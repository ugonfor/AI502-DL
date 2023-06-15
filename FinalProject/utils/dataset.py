import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import PIL

traindir = "/raid/workspace/cvml_user/rhg/dataset/imagenet-1k-train/"
valdir = "/raid/workspace/cvml_user/rhg/dataset/imagenet-1k-val/"

def cifar10dataset():
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=8)
    
    return trainloader, testloader

def ImagenetValDataset(batch_size=32, num_workers=32):

    normalize = transforms.Normalize(0.5, 0.5)
    print('Using image size', 224)
    
    val_transforms = transforms.Compose([
        transforms.Resize(224, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_transforms),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    return val_loader

def ImagenetTrainDataset(batch_size=32, num_workers=32):

    normalize = transforms.Normalize(0.5, 0.5)
    print('Using image size', 224)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, train_transforms),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    
    return train_loader


if __name__ == "__main__":
    t1 = ImagenetValDataset()
    for batch in t1:
        print(batch)
        break