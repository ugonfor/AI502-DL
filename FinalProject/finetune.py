import torch
import timm
import time
import shutil
# https://github.com/lukemelas/PyTorch-Pretrained-ViT/blob/master/examples/imagenet/main.py

from utils.dataset import ImagenetValDataset, ImagenetTrainDataset
from patch_model import patch_vit_fix_block_num_confidence




def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def train_model(model, train_loader, criterion, optimizer, epoch):
    """
    train the model
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))
    
    model.train()

    end=time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)

    

def train_classifier_from_scratch(threshold=0.1):
    """
    train the classifier from scratch
    """
    model_name = "vit_base_patch16_224"
    model: timm.models.vision_transformer.VisionTransformer = timm.create_model(model_name, pretrained=True)
    model._init_weights(model.head)
    model.to(device)


    # freeze all module except classifier
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    # patch the classifier
    p_model_3 = patch_vit_fix_block_num_confidence(model, num_blocks=3, threshold=threshold)
    p_model_6 = patch_vit_fix_block_num_confidence(model, num_blocks=6, threshold=threshold)
    p_model_9 = patch_vit_fix_block_num_confidence(model, num_blocks=9, threshold=threshold)
    
    # config
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer1 = torch.optim.SGD(p_model_3.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer2 = torch.optim.SGD(p_model_6.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer3 = torch.optim.SGD(p_model_9.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer4 = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    print("ISSUE!!! FIX THIS!!! Val->Train")
    train_loader = ImagenetValDataset(batch_size=64, num_workers=32)
    val_loader = ImagenetValDataset(batch_size=64, num_workers=32)

    for epoch in range(10):
        # train model
        if Exp_N == 1: train_model(p_model_3, train_loader, criterion, optimizer1, epoch)
        if Exp_N == 2: train_model(p_model_6, train_loader, criterion, optimizer2, epoch)
        if Exp_N == 3: train_model(p_model_9, train_loader, criterion, optimizer3, epoch)
        if Exp_N == 4: train_model(model, train_loader, criterion, optimizer4, epoch)

        # evaluate on validation set
        if Exp_N == 1: validate(val_loader, model, criterion)
        if Exp_N == 2: validate(val_loader, p_model_3, criterion)
        if Exp_N == 3: validate(val_loader, p_model_3, criterion)
        if Exp_N == 4: validate(val_loader, p_model_3, criterion)



    if Exp_N == 1: save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        # 'best_acc1': best_acc1,
        # 'optimizer' : optimizer.state_dict(),
    }, is_best=False, filename='model_scratch.pth.tar')
    if Exp_N == 2: save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': p_model_3.state_dict(),
        # 'best_acc1': best_acc1,
        # 'optimizer' : optimizer.state_dict(),
    }, is_best=False, filename='p_model_3_scratch.pth.tar')
    if Exp_N == 3: save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': p_model_6.state_dict(),
        # 'best_acc1': best_acc1,
        # 'optimizer' : optimizer.state_dict(),
    }, is_best=False, filename='p_model_6_scratch.pth.tar')
    if Exp_N == 4: save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': p_model_9.state_dict(),
        # 'best_acc1': best_acc1,
        # 'optimizer' : optimizer.state_dict(),
    }, is_best=False, filename='p_model_9_scratch.pth.tar')



def finetune_classifier(threshold=0.1):
    """
    finetune the pretrained vit model
    load the weight from timm
    
    freeze all module except classifier

    1. train classifier from scratch
    2. train classifier with pretrained weight

    """

    model_name = "vit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=True)
    model.to(device)

    # freeze all module except classifier
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    # patch the classifier
    p_model_9 = patch_vit_fix_block_num_confidence(model, num_blocks=9, threshold=threshold)
    p_model_6 = patch_vit_fix_block_num_confidence(model, num_blocks=6, threshold=threshold)
    p_model_3 = patch_vit_fix_block_num_confidence(model, num_blocks=3, threshold=threshold)

    # config
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer1 = torch.optim.SGD(p_model_9.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer2 = torch.optim.SGD(p_model_6.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer3 = torch.optim.SGD(p_model_3.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    optimizer4 = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    
    print("ISSUE!!! FIX THIS!!! Val->Train")
    train_loader = ImagenetValDataset(batch_size=32, num_workers=32)
    val_loader = ImagenetValDataset(batch_size=32, num_workers=32)

    for epoch in range(10):
        # train model
        if Exp_N == 5: train_model(p_model_9, train_loader, criterion, optimizer1, epoch)
        if Exp_N == 6: train_model(p_model_6, train_loader, criterion, optimizer2, epoch)
        if Exp_N == 7: train_model(p_model_3, train_loader, criterion, optimizer3, epoch)
        if Exp_N == 8: train_model(model, train_loader, criterion, optimizer4, epoch)
        
        # evaluate on validation set
        if Exp_N == 5: validate(val_loader, model, criterion)
        if Exp_N == 6: validate(val_loader, p_model_3, criterion)
        if Exp_N == 7: validate(val_loader, p_model_3, criterion)
        if Exp_N == 8: validate(val_loader, p_model_3, criterion)

    if Exp_N == 5: save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        # 'best_acc1': best_acc1,
        # 'optimizer' : optimizer.state_dict(),
    }, is_best=False, filename='model_finetune.pth.tar')
    if Exp_N == 6: save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': p_model_3.state_dict(),
        # 'best_acc1': best_acc1,
        # 'optimizer' : optimizer.state_dict(),
    }, is_best=False, filename='p_model_3_finetune.pth.tar')
    if Exp_N == 7: save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': p_model_6.state_dict(),
        # 'best_acc1': best_acc1,
        # 'optimizer' : optimizer.state_dict(),
    }, is_best=False, filename='p_model_6_finetune.pth.tar')
    if Exp_N == 8: save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': p_model_9.state_dict(),
        # 'best_acc1': best_acc1,
        # 'optimizer' : optimizer.state_dict(),
    }, is_best=False, filename='p_model_9_finetune.pth.tar')



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
import argparse
def init_args():
    save_path = "/raid/workspace/cvml_user/rhg/AI502-DL/FinalProject/weights"
    device = 'cuda:0'
    print_freq = 100
    Exp_N = 0

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--save_path', default=save_path, type=str, help='path to save checkpoint')
    parser.add_argument('--device', default=device, type=str, help='device')
    parser.add_argument('--print_freq', default=print_freq, type=int, help='print frequency')
    parser.add_argument('--Exp_N', default=Exp_N, type=int, help='Experiment Number')
    
    args = parser.parse_args()
    return args


def main():
    global save_path, device, print_freq, Exp_N
    args = init_args()
    save_path = args.save_path
    device = args.device
    print_freq = args.print_freq
    Exp_N = args.Exp_N

    finetune_classifier()
    train_classifier_from_scratch()

if __name__ == "__main__":
    main()