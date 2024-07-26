# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import Any, Union
from functools import partial
from torch import nn
from monai.inferers import sliding_window_inference
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training_cvt
from utils.data_utils import get_loader

import warnings
warnings.filterwarnings("ignore")


import numpy as np
import argparse
import wandb
import math
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data.distributed

from utils.seed import fix_seed

"""load model file"""
import networks.CVT3D_Model_Original as CVT3D_Model_Original
import networks.CVT3D_Model_ADPT as CVT3D_Model_ADPT
import networks.CVT3D_Model_LoRA as CVT3D_Model_LoRA
import networks.CVT3D_Model_Prompt as CVT3D_Model_Prompt
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=True, help="start training from saved checkpoint")
parser.add_argument("--data_dir", default="", type=str, help="dataset directory")
parser.add_argument("--json_list", default=".json", type=str, help="dataset json file")

parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--resume_ckpt", action="store_true", help="start training from saved checkpoint")
parser.add_argument("--logdir", default="", type=str, help="directory to save the checkpoint")
parser.add_argument("--logdir_ep", default="", type=str, help="directory to save the checkpoint for each epochs")

parser.add_argument("--tuning", action="store_true", help="fine-tuning from pretrained checkpoint")
parser.add_argument("--pretrained_model_name", default="", type=str, help="pretrained model checkpoint name")
parser.add_argument("--pretrained_dir", default="", type=str, help="pretrained checkpoint directory")

parser.add_argument("--filename", default="", type=str)
parser.add_argument("--wandb_project", default="", type=str)

parser.add_argument("--max_epochs", default="", type=int, help="max number of training epochs")    
parser.add_argument("--val_every", default="", type=int, help="validation frequency")
parser.add_argument("--batch_size", default="", type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")

parser.add_argument("--optim_lr", default="", type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adam", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default="", type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")

parser.add_argument("--lrschedule", default="", type=str, help="type of learning rate scheduler")
parser.add_argument("--warmup_epochs", default="", type=int, help="number of warmup epochs")
parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")

parser.add_argument("--device_name", default="0", type=str, help="number of GPU device to use")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")

parser.add_argument("--workers", default=8, type=int, help="number of workers")

parser.add_argument("--space_x", default=1.21875, type=float, help="voxel spacing in x direction") 
parser.add_argument("--space_y", default=1.21875, type=float, help="voxel spacing in y direction")
parser.add_argument("--space_z", default=1.21875, type=float, help="voxel spacing in z direction")

"""dimension (128, 128, 63) : roi (64, 64, 32)"""
parser.add_argument("--roi_x", default=64, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=64, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")

parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")

### peft module argparser ###
parser.add_argument("--tune_mode", default="", type=str) #tuning layer

parser.add_argument("--rf", default=0, type=int, help="reduction factor of adapter block") # ADPT(Adapters)
parser.add_argument("--r", default=0, type=int, help="rank of lora") # LoRA
parser.add_argument("--lora_alpha", default=0, type=int, help="alpha of lora") #LoRA 

# visual prompt tuning
parser.add_argument("--en1_tokens", default=0, type=int, help="number of prompt embeddings prepended to en1 layer")
parser.add_argument("--en2_tokens", default=0, type=int, help="number of prompt embeddings prepended to en2 layer")
parser.add_argument("--en3_tokens", default=0, type=int, help="number of prompt embeddings prepended to en3 layer")
parser.add_argument("--de1_tokens", default=0, type=int, help="number of prompt embeddings prepended to de1_up layer")
parser.add_argument("--de2_tokens", default=0, type=int, help="number of prompt embeddings prepended to de2_up layer")
parser.add_argument("--conv3_tokens", default=0, type=int, help="number of prompt embeddings prepended to conv3 layer")
parser.add_argument("--conv4_tokens", default=0, type=int, help="number of prompt embeddings prepended to conv4 layer")
parser.add_argument("--deep", action="store_true", help="allow VPT Deep mode") 

parser.add_argument("--num_seed", default=42, type=int, help="number of seed") 
parser.add_argument("--csv_dir", default="", type=str, help="directory to save validation result")
parser.add_argument("--param_name", action="store_true", help="start distributed training")

def main():
    args = parser.parse_args()
    fix_seed(seed_num=args.num_seed)
    wandb.init(project=f'{args.wandb_project}', name=f'{args.filename}', entity="")
    wandb.config.update(args)

    args.amp = not args.noamp
    args.logdir = args.logdir + args.filename
    args.logdir_ep = args.logdir_ep + args.filename
    
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    if not os.path.exists(args.logdir_ep):
        os.makedirs(args.logdir_ep)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):  
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    
    if args.tune_mode in ["adpt"]:
        modelg = CVT3D_Model_ADPT.Generator(reduction_factor=args.rf)
        modeld = CVT3D_Model_ADPT.Discriminator(reduction_factor=args.rf)
    elif args.tune_mode in ["lora"]:
        modelg = CVT3D_Model_LoRA.Generator(rank=args.r, lora_alpha=args.lora_alpha)
        modeld = CVT3D_Model_LoRA.Discriminator(rank=args.r, lora_alpha=args.lora_alpha)
        
    elif args.tune_mode in ["shallow","deep"]: 
        if args.tune_mode == "deep":
            args.deep = True
        else:
            args.deep = False
        modelg = CVT3D_Model_Prompt.Generator(
            en1_tokens=args.en1_tokens, en2_tokens=args.en2_tokens, en3_tokens=args.en3_tokens,
            de1_tokens=args.de1_tokens, de2_tokens=args.de2_tokens, deep=args.deep)
        modeld = CVT3D_Model_Prompt.Discriminator(conv3_tokens=args.conv3_tokens, conv4_tokens=args.conv4_tokens)
    else:
        modelg = CVT3D_Model_Original.Generator()
        modeld = CVT3D_Model_Original.Discriminator()

    loader = get_loader(args)
    loss_func = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    # load ckpt
    if args.tuning:
        model_dict = torch.load(os.path.join(args.pretrained_dir, args.pretrained_model_name))
        if args.tune_mode in ["shallow", "deep"]:
            state_dict = modeld.load_from(model_dict['state_dict_G']) 
            modeld.load_state_dict(state_dict, strict=False)
            state_dict_G = modelg.load_from(model_dict['state_dict'])
            modelg.load_state_dict(state_dict_G, strict=False)
        else:
            modeld.load_state_dict(model_dict['state_dict_G'], strict=False)
            modelg.load_state_dict(model_dict['state_dict'], strict=False)
        print("Use pretrained weights")
            
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=modelg,
        overlap=args.infer_overlap,
    )
    

    pytorch_total_params = sum(p.numel() for p in modelg.parameters() if p.requires_grad)
    print("Total parameters count MODEL_G", pytorch_total_params)
    
    pytorch_total_params = sum(p.numel() for p in modeld.parameters() if p.requires_grad)
    print("Total parameters count MODEL_D", pytorch_total_params)
    best_acc = 0
    start_epoch = 0
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # modelg = nn.DataParallel(modelg)
        # modeld = nn.DataParallel(modeld)
        
    modelg.to(device)
    modeld.to(device)
    
    modelg.cuda(args.gpu)
    modeld.cuda(args.gpu)
        
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(modeld.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        optimizer_G = torch.optim.Adam(modelg.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(modeld.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
        optimizer_G = torch.optim.AdamW(modelg.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(modeld.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight)
        optimizer_G = torch.optim.SGD(modelg.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight)
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,  warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)
        scheduler_G = LinearWarmupCosineAnnealingLR(optimizer_G,  warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=args.max_epochs)
        if args.checkpoint is not None:
                scheduler.step(epoch=start_epoch)
                scheduler_G.step(epoch=start_epoch)
    else:
        scheduler = None
        scheduler_G = None
    
    mse = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    

    accuracy = run_training_cvt(
        modeld=modeld,
        modelg=modelg,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        optimizer_G=optimizer_G,
        loss_func = mse,
        model_inferer=model_inferer,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
    )
    
    return accuracy


if __name__ == "__main__":
    main()










