'''

Todo List:

1. Change the Block Number
2. Random Value vs Dataset
3. CPU vs GPU
4. Use Linear / Fine-tuned Linear / Train Fromscratch Linear

'''

# random seed fix
import numpy as np
import torch
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

import argparse
import timm
import patch_model

import utils.dataset as dataset
from utils.benchmark import benchmark_random, benchmark_dataset

def init_config():
    parser = argparse.ArgumentParser(description="ViT Benchmark")
    return parser

def Exp1_RandomBenchmark(device="cuda:0"):

    print("EXP1: Random Benchmark")
    print(f"Change the Block Number {device}")

    model_name = "vit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=True)

    # for warmup
    baseline = benchmark_random(
        model,
        device=device,
        verbose=False
    )

    # start
    baseline = benchmark_random(
        model,
        device=device,
        verbose=True
    )

    p_model = patch_model.patch_vit_fix_block_num(model, num_blocks=9)
    exp1 = benchmark_random(
        p_model,
        device=device,
        verbose=True
    )
    print(f'improvement: {exp1/baseline:.2f}x')

    p_model = patch_model.patch_vit_fix_block_num(model, num_blocks=6)
    exp2 = benchmark_random(
        p_model,
        device=device,
        verbose=True
    )
    print(f'improvement: {exp2/baseline:.2f}x')

    p_model = patch_model.patch_vit_fix_block_num(model, num_blocks=3)
    exp3 = benchmark_random(
        p_model,
        device=device,
        verbose=True
    )
    print(f'improvement: {exp3/baseline:.2f}x')


def Exp2_DatasetBenchmark(device="cuda:0", threshold=0.1, arg_runs=None, classifier='linear'):
    assert classifier in ['linear', 'finetune', 'scratch']

    print("EXP2: Dataset Benchmark")
    print(f"Exp on imagenet dataset")
    print(f"Change the Block Number")
    print(f'device: {device}')
    print(f'threshold: {threshold}')
    print(f'classifier: {classifier}')

    runs = -1 if device != "cpu" else 50
    if arg_runs != None: runs = arg_runs

    model_name = "vit_base_patch16_224"
    model = timm.create_model(model_name, pretrained=True)
    val_imagenetdataloader = dataset.ImagenetValDataset()
    
    # for warmup
    benchmark_dataset(
        model,
        device=device,
        verbose=False,
        runs=runs,
        dataset=val_imagenetdataloader
    )

    # start
    if classifier == 'linear':
        pass
    elif classifier == 'finetune':
        model.load_state_dict(torch.load('weights/model_finetune.pth.tar')['state_dict'])
    elif classifier == 'scratch':
        model.load_state_dict(torch.load('weights/model_scratch.pth.tar')['state_dict'])

    baseline = benchmark_dataset(
        model,
        device=device,
        verbose=True,
        runs=runs,
        dataset=val_imagenetdataloader
    )

    p_model = patch_model.patch_vit_fix_block_num_confidence(model, num_blocks=9, threshold=threshold)
    if classifier == 'linear':
        pass
    elif classifier == 'finetune':
        p_model.load_state_dict(torch.load('weights/p_model_9_finetune.pth.tar')['state_dict'])
    elif classifier == 'scratch':
        p_model.load_state_dict(torch.load('weights/p_model_9_scratch.pth.tar')['state_dict'])

    exp1 = benchmark_dataset(
        p_model,
        device=device,
        verbose=True,
        runs=runs,
        dataset=val_imagenetdataloader
    )
    print(f'improvement: {exp1/baseline:.2f}x')

    p_model = patch_model.patch_vit_fix_block_num_confidence(model, num_blocks=6, threshold=threshold)
    if classifier == 'linear':
        pass
    elif classifier == 'finetune':
        p_model.load_state_dict(torch.load('weights/p_model_6_finetune.pth.tar')['state_dict'])
    elif classifier == 'scratch':
        p_model.load_state_dict(torch.load('weights/p_model_6_scratch.pth.tar')['state_dict'])    

    exp2 = benchmark_dataset(
        p_model,
        device=device,
        verbose=True,
        runs=runs,
        dataset=val_imagenetdataloader
    )
    print(f'improvement: {exp2/baseline:.2f}x')

    p_model = patch_model.patch_vit_fix_block_num_confidence(model, num_blocks=3, threshold=threshold)
    if classifier == 'linear':
        pass
    elif classifier == 'finetune':
        p_model.load_state_dict(torch.load('weights/p_model_3_finetune.pth.tar')['state_dict'])
    elif classifier == 'scratch':
        p_model.load_state_dict(torch.load('weights/p_model_3_scratch.pth.tar')['state_dict'])

    exp3 = benchmark_dataset(
        p_model,
        device=device,
        verbose=True,
        runs=runs,
        dataset=val_imagenetdataloader
    )
    print(f'improvement: {exp3/baseline:.2f}x')



def main():
    # Random Value / CPU / GPU
    #Exp1_RandomBenchmark('cpu')
    #Exp1_RandomBenchmark('cuda:0')
    
    # Dataset / CPU / GPU / Use Linear / Fine-tuned Linear / Train Fromscratch Linear
    #Exp2_DatasetBenchmark('cpu', threshold=0.1, arg_runs=50, classifier='linear')
    Exp2_DatasetBenchmark('cpu', threshold=0.1, arg_runs=50, classifier='finetune')
    Exp2_DatasetBenchmark('cpu', threshold=0.1, arg_runs=50, classifier='scratch')

    Exp2_DatasetBenchmark('cuda:0', threshold=0.1, classifier='linear')
    Exp2_DatasetBenchmark('cuda:0', threshold=0.1, classifier='finetune')
    Exp2_DatasetBenchmark('cuda:0', threshold=0.1, classifier='scratch')


if __name__ == "__main__":
    main()
    