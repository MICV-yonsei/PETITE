# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
warnings.filterwarnings("ignore")

import argparse
import os

import wandb
import numpy as np
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.nn.functional as F
from torch.nn import MSELoss
import torch.utils.data.distributed

from networks.unetr import UNETR, Prompted_UNETR, Adapted_UNETR, Lora_UNETR, SSF_UNETR
from trainer import run_training
from utils.data_utils import get_loader
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from monai.inferers import sliding_window_inference
from monai.utils import MetricReduction
from monai.metrics import PSNRMetric, RMSEMetric
from utils.seed import fix_seed
parser = argparse.ArgumentParser(description="UNETR + ADNI Reproduce")
# Data
parser.add_argument("--logdir", default="", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--data_dir", default="", type=str, help="dataset directory")
parser.add_argument("--json_list", default="", type=str, help="dataset json file")
parser.add_argument("--pretrained_dir", default="", help="start training from saved checkpoint")
parser.add_argument("--pretrained_model_name", default="", type=str, help="name of pretrained model checkpoint")
# Training
parser.add_argument("--max_epochs", default=200, type=int, help="max number of training epochs")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
parser.add_argument("--batch_size", default=6, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--optim_lr", default=1e-2, type=float, help="optimization learning rate")
parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--reg_weight", default=1e-4, type=float, help="regularization weight")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
parser.add_argument("--lr_gamma", default=0.1, type=float,  help="every lr_step_size epochs, decay learning rate by a factor of lr_gamma")
parser.add_argument("--lr_step_size", default=50, type=int,  help="every lr_step_size epochs, decay learning rate by a factor of lr_gamma")
parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
# Distributed Training
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
# UNETR
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=32, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
# data utils
parser.add_argument("--minv", default=0.0, type=float)
parser.add_argument("--maxv", default=1.0, type=float)
parser.add_argument("--space_x", default=1.21875, type=float, help="spacing in x direction") #voxel spacing
parser.add_argument("--space_y", default=1.21875, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.21875, type=float, help="spacing in z direction") 
parser.add_argument("--roi_x", default=64, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=64, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=64, type=int, help="roi size in z direction")
# inference
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
# SSIM, PSNR
parser.add_argument("--factor", default=None, type=float)
parser.add_argument("--channel_wise", default=False, type=float)
parser.add_argument("--win_size", default=7, type=int, help="gaussian weighting window size")
parser.add_argument("--k1", default=0.01, type=float, help="constant used in the luminance denominator")
parser.add_argument("--k2", default=0.03, type=float, help="constant used in the contrast denominator")
parser.add_argument("--spatial_dims", default=3, type=int, help="if 2, input shape is expected to be (B,C,H,W). if 3, it is expected to be (B,C,H,W,D)")
parser.add_argument("--max_val", default=1.0, type=int)
# PEFT
parser.add_argument("--bias_mode", default="prompt", type=str, help="Tuning mode")
parser.add_argument("--tune_mode", default="prompt", type=str, help="Tuning mode")
parser.add_argument("--num_tokens", default=50, type=int, help="number of prompt tokens")
parser.add_argument("--deep", action="store_true", help="execute deep forward embedding in prompt")
parser.add_argument("--rf", default=8, type=int, help="Reduction factor of adapter")
parser.add_argument("--r", default=8, type=int, help="Rank of LoRA")
parser.add_argument("--lora_alpha", default=1, type=int, help="Alpha value of LoRA")
# Wandb
parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
parser.add_argument("--wandb_project", default="", type=str)

parser.add_argument("--num_seed", default=42, type=int, help="number of seed") 
parser.add_argument("--csv_dir", default="", type=str, help="directory to save validation result")

def main():
    args = parser.parse_args()
    fix_seed(seed_num=args.num_seed)
    
    if args.save_checkpoint:
        print("Save checkpoint and wandb log")
        wandb.init(project=f'{args.wandb_project}', name=f'{args.filename}', entity="pet-ft")
        wandb.config.update(args)
        
    args.amp = not args.noamp
    args.logdir = args.logdir + args.filename
    
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
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
    loader = get_loader(args)
    
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    if args.tune_mode in ["prompt"]:
        model = Prompted_UNETR(
                    in_channels=args.in_channels,
                    out_channels=args.out_channels,
                    img_size=(args.roi_x, args.roi_y, args.roi_z),
                    feature_size=args.feature_size,
                    hidden_size=args.hidden_size,
                    mlp_dim=args.mlp_dim,
                    num_heads=args.num_heads,
                    pos_embed=args.pos_embed,
                    norm_name=args.norm_name,
                    conv_block=True,
                    res_block=True,
                    dropout_rate=args.dropout_rate,
                    num_tokens=args.num_tokens,
                    deep=args.deep
                )

    elif args.tune_mode in ["adpt"]:
        model = Adapted_UNETR(
                    in_channels=args.in_channels,
                    out_channels=args.out_channels,
                    img_size=(args.roi_x, args.roi_y, args.roi_z),
                    feature_size=args.feature_size,
                    hidden_size=args.hidden_size,
                    mlp_dim=args.mlp_dim,
                    num_heads=args.num_heads,
                    pos_embed=args.pos_embed,
                    norm_name=args.norm_name,
                    conv_block=True,
                    res_block=True,
                    dropout_rate=args.dropout_rate,
                    rf=args.rf,
                )
  
    elif args.tune_mode in ["lora"]:
        model = Lora_UNETR(
                    in_channels=args.in_channels,
                    out_channels=args.out_channels,
                    img_size=(args.roi_x, args.roi_y, args.roi_z),
                    feature_size=args.feature_size,
                    hidden_size=args.hidden_size,
                    mlp_dim=args.mlp_dim,
                    num_heads=args.num_heads,
                    pos_embed=args.pos_embed,
                    norm_name=args.norm_name,
                    conv_block=True,
                    res_block=True,
                    dropout_rate=args.dropout_rate,
                    r=args.r,
                    lora_alpha=args.lora_alpha,
                )

    elif args.tune_mode in ["fft", "bn", "ln", "bias", "org", "vit", "vit_top", "vit_down", "vit_skip", "novit", None]:
        model = UNETR(
                    in_channels=args.in_channels,
                    out_channels=args.out_channels,
                    img_size=(args.roi_x, args.roi_y, args.roi_z),
                    feature_size=args.feature_size,
                    hidden_size=args.hidden_size,
                    mlp_dim=args.mlp_dim,
                    num_heads=args.num_heads,
                    pos_embed=args.pos_embed,
                    norm_name=args.norm_name,
                    conv_block=True,
                    res_block=True,
                    dropout_rate=args.dropout_rate,
                )
        
    elif args.tune_mode in ["ssf"]:
            model = SSF_UNETR(
                        in_channels=args.in_channels,
                        out_channels=args.out_channels,
                        img_size=(args.roi_x, args.roi_y, args.roi_z),
                        feature_size=args.feature_size,
                        hidden_size=args.hidden_size,
                        mlp_dim=args.mlp_dim,
                        num_heads=args.num_heads,
                        pos_embed=args.pos_embed,
                        norm_name=args.norm_name,
                        conv_block=True,
                        res_block=True,
                        dropout_rate=args.dropout_rate
                    )
    else:
        raise ValueError(f"Unknown tune mode {args.tune_mode}")

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0


    if args.tune_mode is not None:
        
        model_dict = torch.load(os.path.join(args.pretrained_dir, args.pretrained_model_name), map_location="cpu")
        try:
            model.load_state_dict(model_dict["state_dict"], strict=False)
        except:
            del model_dict["state_dict"]["vit.patch_embedding.position_embeddings"]
            model.load_state_dict(model_dict["state_dict"], strict=False)
        print("Use pretrained weight")

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.tune_mode is not None:
            scheduler.step(epoch=start_epoch)
    elif args.lrschedule == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    else:
        scheduler = None
    
    mse_loss = MSELoss(size_average=None, reduce=None, reduction='mean')
    rmse = RMSEMetric(reduction=MetricReduction.MEAN, get_not_nans=False)
    psnr = PSNRMetric(max_val=args.max_val, reduction=MetricReduction.MEAN, get_not_nans=False)
    
    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=mse_loss,
        metric_func=rmse,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
    )
    
    
    return accuracy


if __name__ == "__main__":
    main()






