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

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
import wandb
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
from torch.cuda.amp import GradScaler, autocast
import nibabel as nib
from monai.utils import MetricReduction
from monai.metrics import RMSEMetric

from utils.utils import AverageMeter, distributed_all_gather
from utils.valid_utils import make_new_df, fill_df
from utils.metric import SSIMMetric
import loralib as lora

def show_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def Tuner(modeld, modelg, args):
    """Fine-tuning Options"""
    modeld.requires_grad_(False)
    modelg.requires_grad_(False)  

    if args.tune_mode == "fft":
        modeld.requires_grad_(True)
        modelg.requires_grad_(True)
        
    elif args.tune_mode == "ln":
        for m in modeld.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        for m in modelg.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
                
    elif args.tune_mode == "bias":
        for n, p in modeld.named_parameters():
            if 'bias' in n:
                p.requires_grad_(True)
        for n, p in modelg.named_parameters():
            if 'bias' in n:
                p.requires_grad_(True)

    elif args.tune_mode == "adpt":
        for n, p in modeld.named_parameters():
            if 'adapter_' in n:
                p.requires_grad_(True)
        for n, p in modelg.named_paramets():
            if 'adapter_' in n:
                p.requires_grad_(True)
                
    elif args.tune_mode == "lora":
        for n, p in modeld.named_parameters():
            if 'lora_' in n:
                p.requires_grad_(True)
        for n, p in modelg.named_parameters():
            if 'lora_' in n:
                p.requires_grad_(True)
                        
    elif args.tune_mode in ["shallow", "deep"]:
        for n, p in modeld.named_parameters():
            if args.roi_z == 64:
                if 'prompt_embeddings1' in n:
                    p.requires_grad_(True)
                if args.deep:   
                    if 'deep_prompt_embeddings1' in n:
                        p.requires_grad_(True)
                    
            elif args.roi_z == 32:
                if 'prompt_embeddings2' in n:
                    p.requires_grad_(True)
                if args.deep:
                    if 'deep_prompt_embeddings2' in n:
                        p.requires_grad_(True)
                    
        for n, p in modelg.named_parameters():
            if args.roi_z == 64:
                if 'prompt_embeddings1' in n:
                    p.requires_grad_(True)
                if args.deep:   
                    if 'deep_prompt_embeddings1' in n:
                        p.requires_grad_(True)
            elif args.roi_z == 32:
                if 'prompt_embeddings2' in n:
                    p.requires_grad_(True)
                if args.deep:
                    if 'deep_prompt_embeddings2' in n:
                        p.requires_grad_(True)

    elif args.tune_mode == "ssf":
        for n, p in modelg.named_parameters():
            if 'ssf_' in n:
                p.requires_grad_(True)
        for n, p in modeld.named_parameters():
            if 'ssf_' in n:
                p.requires_grad_(True)
                             
    else:
        print("None of layers are unfrozen")
            
    return modeld, modelg

def train_epoch(modeld, modelg, loader, optimizer, optimizer_G, epoch, loss_func, args):   
    modeld.train()
    modelg.train() 

    if args.tune_mode:
        modeld, modelg = Tuner(modeld=modeld, modelg=modelg, args=args)

    start_time = time.time()
    
    run_gloss = AverageMeter()
    run_dloss = AverageMeter()
    run_psnr = AverageMeter()
    run_ssim = AverageMeter()
    run_nrmse = AverageMeter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
            
        x, y = data.cuda(args.rank), target.cuda(args.rank)
        for param in modeld.parameters():
            param.grad = None
        for param in modelg.parameters():
            param.grad = None
        
        
        
        with autocast(enabled=args.amp):
            
            BCELoss = nn.BCEWithLogitsLoss()
            L1 = nn.L1Loss()

            # G_train
            modelg.zero_grad()

            G_output = modelg(x) 
            X_fake = torch.cat([x, G_output], dim=1) 
            D_output_f = modeld(X_fake)
            
            G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
            G_L1_Loss = L1(G_output, y) 
            G_loss = G_BCE_loss + 100 * G_L1_Loss

            optimizer_G.zero_grad()
            G_loss.requires_grad_(True)
            G_loss.backward()
            optimizer_G.step()
            
            # D_train
            modeld.zero_grad()

            xy = torch.cat([x, y], dim=1)  
            D_output_r = modeld(xy).squeeze()
            D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device))
            
            G_output = modelg(x)                      
            X_fake = torch.cat([x, G_output], dim=1)    
            D_output_f = modeld(X_fake).squeeze()
            D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))
            D_loss = (D_real_loss + D_fake_loss) * 0.5
            
            optimizer.zero_grad()
            D_loss.requires_grad_(True)
            D_loss.backward()
            optimizer.step()
            
            # Wandb Logging
            loss = loss_func(G_output, y)
            psnr = 10*torch.log10(1.0 / loss)
            data_range = y.max().unsqueeze(0) 
            ssim = SSIMMetric(data_range=data_range,spatial_dims=3)._compute_metric(G_output, y)
            a = RMSEMetric(reduction=MetricReduction.MEAN, get_not_nans=False)    
            rmse = a(G_output, y)
            nrmse = rmse / (y.max()-y.min())   


        if args.distributed:
            G_loss_list = distributed_all_gather([G_loss], out_numpy=True, is_valid=i < loader.sampler.valid_length) 
            D_loss_list = distributed_all_gather([D_loss], out_numpy=True, is_valid=i < loader.sampler.valid_length)
            psnr_list = distributed_all_gather([psnr], out_numpy=True, is_valid=i < loader.sampler.valid_length)
            ssim_list = distributed_all_gather([ssim], out_numpy=True, is_valid=i < loader.sampler.valid_length)
            nrmse_list = distributed_all_gather([nrmse], out_numpy=True, is_valid=i < loader.sampler.valid_length)
            run_gloss.update(np.mean(np.mean(np.stack(G_loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
            run_dloss.update(np.mean(np.mean(np.stack(D_loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
            run_psnr.update(np.mean(np.mean(np.stack(psnr_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
            run_ssim.update(np.mean(np.mean(np.stack(ssim_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
            run_nrmse.update(np.mean(np.mean(np.stack(nrmse_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
        else:
            run_gloss.update(G_loss.item(), n=args.batch_size)
            run_dloss.update(D_loss.item(), n=args.batch_size)
            run_psnr.update(np.mean(psnr), n=args.batch_size)
            run_ssim.update(np.mean(ssim), n=args.batch_size)
            run_nrmse.update(np.mean(nrmse), n=args.batch_size)

        if args.rank == 0:     
            print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, i, len(loader)),
                    "G_loss: {:.4f}".format(run_gloss.avg),
                    "D_loss : {:.4f}".format(run_dloss.avg),
                    "PSNR: {:.4f}".format(run_psnr.avg),
                    "SSIM: {:.4f}".format(run_ssim.avg),
                    "NRMSE: {:.4f}".format(run_nrmse.avg)   
                    )
            
    for param in modelg.parameters():
        param.grad=None
    for param in modeld.parameters():
        param.grad=None

    if args.save_checkpoint:
        wandb.log({
                    "G_loss" : run_gloss.avg, "D_loss" : run_dloss.avg, 
                    "PSNR" : run_psnr.avg, "SSIM" : run_ssim.avg, "NRMSE" : run_nrmse.avg,
                    "Epoch": epoch
                })        


def val_epoch(modeld, modelg, loader, optimizer, optimizer_G, epoch, loss_func, args, model_inferer=None, modeld_inferer=None,): 
    modeld.eval()
    modelg.eval()
    modeld.requires_grad_(False)
    modelg.requires_grad_(False)
    
    start_time = time.time()

    run_gloss = AverageMeter()
    run_dloss = AverageMeter()
    run_psnr = AverageMeter()
    run_ssim = AverageMeter()
    run_nrmse = AverageMeter()
    
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data, target = data.cuda(args.rank), target.cuda(args.rank)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            x = data.to(device)   
            y = target.to(device)  

            
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    G_output = model_inferer(data)

                else:
                    G_output = modelg(data)

                BCELoss = nn.BCEWithLogitsLoss()
                L1 = nn.L1Loss()

                # G_val
                X_fake = torch.cat([data, G_output], dim=1) 

                if modeld_inferer is not None:
                    D_output_f = modeld_inferer(X_fake)
                else:
                    D_output_f = modeld(X_fake)

                G_BCE_loss = BCELoss(D_output_f, torch.ones(D_output_f.size()).to(device))
                G_L1_Loss = L1(G_output, target) 
                G_loss = G_BCE_loss + 100 * G_L1_Loss

                # D_val
                xy = torch.cat([x, y], dim=1)  

                if modeld_inferer is not None:
                    D_output_r = modeld_inferer(xy).squeeze()
                else:
                    D_output_r = modeld(xy).squeeze()

                D_real_loss = BCELoss(D_output_r, torch.ones(D_output_r.size()).to(device))
                
                X_fake = torch.cat([x, G_output], dim=1)    
                if modeld_inferer is not None:
                    D_output_f = modeld_inferer(X_fake).squeeze()
                else:
                    D_output_f = modeld(X_fake).squeeze()

                D_fake_loss = BCELoss(D_output_f, torch.zeros(D_output_f.size()).to(device))
                D_loss = (D_real_loss + D_fake_loss) * 0.5

                # Wandb logging
                loss = loss_func(G_output, y)
                psnr = 10*torch.log10(1.0 / loss)
                data_range = target.max().unsqueeze(0) 
                ssim = SSIMMetric(data_range=data_range,spatial_dims=3)._compute_metric(G_output, target)
                a = RMSEMetric(reduction=MetricReduction.MEAN, get_not_nans=False)    
                rmse = a(G_output, target)
                nrmse = rmse / (target.max()-target.min())   
                
            if args.distributed:
                G_loss_list = distributed_all_gather([G_loss], out_numpy=True, is_valid=i < loader.sampler.valid_length) 
                D_loss_list = distributed_all_gather([D_loss], out_numpy=True, is_valid=i < loader.sampler.valid_length)
                psnr_list = distributed_all_gather([psnr], out_numpy=True, is_valid=i < loader.sampler.valid_length)
                ssim_list = distributed_all_gather([ssim], out_numpy=True, is_valid=i < loader.sampler.valid_length)
                nrmse_list = distributed_all_gather([nrmse], out_numpy=True, is_valid=i < loader.sampler.valid_length)
                run_gloss.update(np.mean(np.mean(np.stack(G_loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
                run_dloss.update(np.mean(np.mean(np.stack(D_loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
                run_psnr.update(np.mean(np.mean(np.stack(psnr_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
                run_ssim.update(np.mean(np.mean(np.stack(ssim_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
                run_nrmse.update(np.mean(np.mean(np.stack(nrmse_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size) 
            else:
                run_gloss.update(G_loss.item(), n=args.batch_size)
                run_dloss.update(D_loss.item(), n=args.batch_size)
                run_psnr.update(np.mean(psnr), n=args.batch_size)
                run_ssim.update(np.mean(ssim), n=args.batch_size)
                run_nrmse.update(np.mean(nrmse), n=args.batch_size)
                        
            if args.rank == 0:
                print(
                        "Val {}/{} {}/{}".format(epoch, args.max_epochs, i, len(loader)),
                        "Val_G_loss: {:.4f}".format(G_loss.item()),
                        "Val_D_loss : {:.4f}".format(D_loss.item()),
                        "Val_PSNR: {:.4f}".format(np.mean(psnr)),
                        "Val_SSIM: {:.4f}".format(np.mean(ssim)),
                        "Val_NRMSE: {:.4f}".format(np.mean(nrmse))     
                        )
    if args.rank == 0:
        print(  
                "Val Average",
                "Val_G_loss: {:.4f}".format(run_gloss.avg),
                "Val_D_loss : {:.4f}".format(run_dloss.avg),
                "Val_PSNR: {:.4f}".format(run_psnr.avg),
                "Val_SSIM: {:.4f}".format(run_ssim.avg),
                "Val_NRMSE: {:.4f}".format(run_nrmse.avg)     
                )

    if args.save_checkpoint:
        wandb.log({
                    "Val_Gloss" : run_gloss.avg, "Val_Dloss" : run_dloss.avg, "Val_PSNR" : run_psnr.avg, "Val_SSIM" : run_ssim.avg, "Val_NRMSE" : run_nrmse.avg, "Epoch": epoch
                
                }) 

    return [run_psnr.avg, run_ssim.avg, run_nrmse.avg]



def save_checkpoint(modeld, modelg, epoch, args, filename, best_acc=0, optimizer=None, optimizer_G=None, scheduler=None, scheduler_G=None):
    if not args.tuning:
        state_dict = modeld.state_dict() if not args.distributed else modeld.module.state_dict()
        state_dict_G = modelg.state_dict() if not args.distributed else modelg.module.state_dict()
    else:
        modeld, modelg = Tuner(modeld=modeld, modelg=modelg, args=args)
        state_dict = {name: param for name, param in modeld.named_parameters() if param.requires_grad}
        state_dict_G = {name: param for name, param in modelg.named_parameters() if param.requires_grad}
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict, "state_dict_G": state_dict_G}
    
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if optimizer_G is not None:
        save_dict["optimizer_G"] = optimizer_G.state_dict()      
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if scheduler_G is not None:
        save_dict["scheduler_G"] = scheduler_G.state_dict()
        
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def run_training_cvt(
    modeld,
    modelg,
    train_loader,
    val_loader,
    optimizer,
    optimizer_G,
    loss_func,
    args,
    model_inferer=None,
    modeld_inferer=None,
    scheduler=None,
    scheduler_G=None,
    start_epoch=0,
    ):

    writer = None
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
            modeld, modelg, train_loader, optimizer, optimizer_G, loss_func=loss_func, epoch=epoch,  args=args, 
        )
        
        if args.rank == 0:
                print(
                    "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                    "time {:.2f}s".format(time.time() - epoch_time),
                ) 
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)

        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc = val_epoch(
                modeld=modeld,
                modelg=modelg,
                loader=val_loader,
                optimizer=optimizer,
                optimizer_G=optimizer_G,
                epoch=epoch,
                loss_func=loss_func,
                model_inferer=model_inferer,
                modeld_inferer=modeld_inferer,
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
            if args.rank == 0 and args.save_checkpoint:
                save_checkpoint(modeld=modeld, modelg=modelg, epoch=epoch, args=args, best_acc=val_acc_max, filename=f"final_{args.filename}.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, f"final_{args.filename}.pt"), os.path.join(args.logdir, f"best_{args.filename}.pt"))

        if scheduler is not None:
            scheduler.step()
        if scheduler_G is not None:
            scheduler_G.step()

    print("Training Finished ><!")

    return val_acc_max

