
# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR                 CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import time
import wandb
import shutil
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast

from utils.metric import SSIMMetric
from utils.utils import AverageMeter, distributed_all_gather
from utils.valid_utils import make_new_df, fill_df

def Tuner(model, args):
    """Fine-tuning Options"""
    model.requires_grad_(False)

    if args.tune_mode == "fft":
        model.requires_grad_(True)
        
    elif args.tune_mode == "last":
        for name, param in model.named_parameters():
            if 'decoder2' in name:
                param.requires_grad_(True)
                
    elif args.tune_mode == "vit":
        for name, param in model.named_parameters():
            if 'vit' in name:
                param.requires_grad_(True)
    elif args.tune_mode == "skip":
        for name, param in model.named_parameters():
            if 'vit.blocks.2' in name:
                param.requires_grad_(True)
            elif 'vit.blocks.5' in name:
                param.requires_grad_(True)
            elif 'vit.blocks.8' in name:
                param.requires_grad_(True)
            elif 'vit.blocks.11' in name:
                param.requires_grad_(True)
    elif args.tune_mode == "connection":
        for name, param in model.named_parameters():
            if 'encoder' in name:
                param.requires_grad_(True)
    elif args.tune_mode == "novit":
        for name, param in model.named_parameters():
            if 'decoder' not in name:
                param.requires_grad_(True)
                
    elif args.tune_mode == "ln":
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)

    elif args.tune_mode == "bias":
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad_(True)

    elif args.tune_mode == "adpt":
        for name, param in model.named_parameters():
            if 'adapter_' in name:
                param.requires_grad_(True)
                
    elif args.tune_mode == "lora":
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad_(True)
                
    elif args.tune_mode == "prompt_":
        for name, param in model.named_parameters():
            if "vit.prompt_embeddings" in name :
                param.requires_grad_(True)     
            if args.deep:
                if "vit.deep_prompt_embeddings" in name :
                    param.requires_grad_(True)                 
                    
    elif args.tune_mode == "ssf" :
        for name, param in model.named_parameters():
            if "ssf_" in name :
                param.requires_grad_(True)                            
    
    
    
    else:
        raise ValueError(f"None of layers requires gradient: MODE {args.tune_mode}")

    return model

def show_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args, metric_func):
    model.train()

    if args.tune_mode:
        model = Tuner(model=model, args=args)

    start_time = time.time()
    
    run_loss = AverageMeter()
    run_psnr = AverageMeter()
    run_ssim = AverageMeter()
    run_nrmse = AverageMeter()

    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
            
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        
        for param in model.parameters():
            param.grad = None
            
        with autocast(enabled=args.amp):
            logits = model(data)   
            loss = loss_func(logits, target)

            rmse = metric_func(logits, target)
            nrmse = rmse / (target.max()-target.min())  
            psnr = 20*math.log10(args.max_val) - 10*torch.log10(loss)
            data_range = target.max().unsqueeze(0) #gt.max().squeeze() #range max 1
            ssim = SSIMMetric(data_range=data_range,spatial_dims=3)._compute_metric(logits, target)
            
            optimizer.zero_grad() 
    
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward() 
            optimizer.step() 
        
            
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            psnr_list = distributed_all_gather([psnr], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            ssim_list = distributed_all_gather([ssim], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            nrmse_list = distributed_all_gather([nrmse], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size)
            run_psnr.update(np.mean(np.mean(np.stack(psnr_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
            run_ssim.update(np.mean(np.mean(np.stack(ssim_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
            run_nrmse.update(np.mean(np.mean(np.stack(nrmse_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            run_psnr.update(np.mean(psnr), n=args.batch_size)
            run_ssim.update(np.mean(ssim), n=args.batch_size)
            run_nrmse.update(np.mean(nrmse), n=args.batch_size)

        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "mseLoss: {:.4f}".format(run_loss.avg),
                "PSNR: {:.4f}".format(run_psnr.avg),
                "SSIM: {:.4f}".format(run_ssim.avg),
                "NRMSE: {:.4f}".format(run_nrmse.avg)     
                )   
        start_time = time.time()
            
    for param in model.parameters():
        param.grad=None

    if args.save_checkpoint:    
        wandb.log({"Epoch" : epoch, "mseLoss" : run_loss.avg, "PSNR" : run_psnr.avg, "SSIM" : run_ssim.avg, "NRMSE" : run_nrmse.avg })
                 
def val_epoch(model, loader, epoch, optimizer, scaler, loss_func, args, model_inferer, metric_func, post_label=None, post_pred=None):
    model.eval()

    start_time = time.time()

    run_loss= AverageMeter()
    run_psnr = AverageMeter()
    run_ssim = AverageMeter()
    run_nrmse = AverageMeter()

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data) 
                else:
                    logits = model(data)
                        
                loss = loss_func(logits, target)
                rmse = metric_func(logits, target)                
                nrmse = rmse / (target.max()-target.min())
                psnr = 20*math.log10(args.max_val) - 10*torch.log10(loss)                
                data_range = target.max().unsqueeze(0)
                ssim = SSIMMetric(data_range=data_range,spatial_dims=3)._compute_metric(logits, target)        
                
            if args.distributed:
                loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                psnr_list = distributed_all_gather([psnr], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                ssim_list = distributed_all_gather([ssim], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                nrmse_list = distributed_all_gather([nrmse], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
                run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size)
                run_psnr.update(np.mean(np.mean(np.stack(psnr_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
                run_ssim.update(np.mean(np.mean(np.stack(ssim_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
                run_nrmse.update(np.mean(np.mean(np.stack(nrmse_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
            else:
                run_loss.update(loss.item(), n=args.batch_size)
                run_psnr.update(np.mean(psnr), n=args.batch_size)
                run_ssim.update(np.mean(ssim), n=args.batch_size)
                run_nrmse.update(np.mean(nrmse), n=args.batch_size)
            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "val_mseLoss: {:.4f}".format(loss.item()),
                    "val_PSNR: {:.4f}".format(np.mean(psnr)),
                    "val_RMSE: {:4f}".format(np.mean(rmse)),
                    "val_SSIM : {:.4f}".format(np.mean(ssim)),
                    "val_NRMSE : {:.4f}".format(np.mean(nrmse)),
                )
                
            start_time = time.time()

    if args.rank == 0:
        print(  
                "Val Average",
                "Val_mseLoss: {:.4f}".format(run_loss.avg),
                "Val_PSNR: {:.4f}".format(run_psnr.avg),
                "Val_SSIM: {:.4f}".format(run_ssim.avg),
                "Val_NRMSE: {:.4f}".format(run_nrmse.avg)     
                )

    if args.save_checkpoint:
        wandb.log({"val_PSNR" : run_psnr.avg, "val_SSIM" : run_ssim.avg, "val_NRMSE" : run_nrmse.avg,"val_run_loss" : run_loss.avg})

    return [run_psnr.avg, run_ssim.avg, run_nrmse.avg]# np.mean(avg_acc) #run_loss.avg
   
def save_checkpoint(model, epoch, args, filename, best_acc=0, optimizer=None, scheduler=None):
    if not args.tune_mode:
        state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    else:
        model = Tuner(model=model, args=args)
        state_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
            
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    
    
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
        
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    metric_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
):
    scaler = None
    if args.amp:
        scaler = GradScaler()

    # Save result as csv
    folder_dir = os.path.dirname(args.csv_dir)
    if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
    if not os.path.exists(args.csv_dir):
        df = make_new_df(max_epoch=args.max_epochs, val_every=args.val_every)
    else:
        df = {
            '1': pd.read_excel(args.csv_dir, sheet_name='1', header=0, index_col=0),
            '2': pd.read_excel(args.csv_dir, sheet_name='2', header=0, index_col=0),
            '3': pd.read_excel(args.csv_dir, sheet_name='3', header=0, index_col=0),
            }     


    val_acc_max = [0.0, 0.0, 1.0]
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, metric_func=metric_func,loss_func=loss_func, args=args
        )

        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            show_params(model)
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                loss_func=loss_func,
                metric_func=metric_func,
                scaler=scaler,
                optimizer=optimizer,
                model_inferer=model_inferer,
                args=args,
            )

            if args.rank == 0:
                prev_psnr, prev_ssim, prev_nrmse = val_acc_max[0],val_acc_max[1],val_acc_max[2]
                psnr, ssim, nrmse = val_acc[0],val_acc[1],val_acc[2]
                df = fill_df(df, psnr, ssim, nrmse, time.time()-epoch_time, epoch, args)
                print("Val prev best PSNR: {:.6f} SSIM {:.6f} NRMSE {:.6f}".format(prev_psnr, prev_ssim, prev_nrmse))
                print("Val new PSNR: {:.6f} SSIM {:.6f} NRMSE {:.6f}".format(psnr, ssim, nrmse))
                if (prev_psnr<psnr) and (prev_ssim<ssim) and (prev_nrmse>nrmse):
                    val_acc_max = val_acc
                    b_new_best = True
                elif (prev_ssim<ssim):
                    if (prev_psnr<psnr) and (prev_nrmse<nrmse):
                        val_acc_max = val_acc
                        b_new_best = True
                    elif (prev_psnr>psnr) and (prev_nrmse>nrmse):
                        val_acc_max = val_acc
                        b_new_best = True
                else:
                    b_new_best = False
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename=f"final_{args.filename}.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, f"final_{args.filename}.pt"), os.path.join(args.logdir, f"{args.filename}.pt"))

        if scheduler is not None:
            scheduler.step()

    print("Training Finished ><!")

    return val_acc_max

