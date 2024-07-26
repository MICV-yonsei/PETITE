from tqdm import tqdm
import pandas as pd
import argparse
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", default="/home/work/kochanha/yumin/UNETR/BTCV/VPT_ckpt_log/runs_ckpt/base_raw", type=str, help="")
parser.add_argument("--new_ckpt_dir", default="/home/work/kochanha/gayoon/CVTGAN/VPT_ckpt_log/runs_ckpt/adpt/rf8_re/", type=str, help="")
parser.add_argument("--tune_mode", default="adpt", type=str, help="")
parser.add_argument("--csv_dir", default="/home/work/kochanha/gayoon/UNETR/runs_ckpt/ckpt_raw.csv", type=str, help="")


def find_best(args):
    paths = glob.glob(os.path.join(args.ckpt_dir, "*/*.pt"))
    df = pd.DataFrame({'checkpoint':[],'epoch':[]})
    
    for i in tqdm(range(len(paths))):
        path_name = paths[i].split("/")[-1].split(".")[0]
        if 'final' in path_name: 
            pass
        else:
            ckpt = torch.load(paths[i])
            df.loc[i] = [path_name, ckpt['epoch']]

    df.to_csv(args.csv_dir, header=True) 
    return df

def change_dict(args):
    paths = glob.glob(os.path.join(args.ckpt_dir, "*/*.pt"))

    if args.tune_mode == "adpt":
        label = "adapter_"
    elif args.tune_mode == "lora":
        label = "lora_"

    for i in tqdm(range(len(paths))):
        ckpt = torch.load(paths[i])
        ckpt_name = paths[i].split('/')[-1]
        folder_dir = '/'.join([args.new_ckpt_dir,paths[i].split('/')[-2]])
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)

        print(paths[i])
        state_dict_G = {key: value for key, value in ckpt['state_dict_G'].items() if label in key}
        new_ckpt = {"epoch": ckpt['epoch'], "best_acc": ckpt['best_acc'], "state_dict_G": state_dict_G}
        torch.save(new_ckpt, f"{folder_dir}/{ckpt_name}")
        print(state_dict_G.keys())

if __name__ == '__main__':
    args = parser.parse_args()

    # change_dict(args)
    best = find_best(args)
    print(best['epoch'].value_counts())
    