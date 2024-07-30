from monai.metrics.regression import SSIMMetric
from einops import rearrange
from monai import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import argparse
import torch
import glob
import json
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser()
parser.add_argument("--data_json", default="/root_dir/ADNI/Dynamic/Resolution/Resolution.json", type=str, help="data split file path")
parser.add_argument("--image_path", default="/root_dir/ADNI/Dynamic/Resolution/images/", type=str, help="image folder path (Input)")
parser.add_argument("--label_path", default="/root_dir/ADNI/Dynamic/Resolution/labels/", type=str, help="label folder path (GT)")

parser.add_argument("--base_path", default="", type=str, help="base model results folder path (CVTGAN)")
parser.add_argument("--fft_path", default="", type=str, help="full fine tuning results folder path")
parser.add_argument("--ln_path", default="", type=str, help="layernorm results folder path")
parser.add_argument("--bias_path", default="", type=str, help="bitfit results folder path")
parser.add_argument("--lora_path", default="", type=str, help="lora results folder path")
parser.add_argument("--adapter_path", default="", type=str, help="adapters results folder path")
parser.add_argument("--ssf_path", default="", type=str, help="ssf results folder path")
parser.add_argument("--shallow_path", default="", type=str, help="VPT-shallow results folder path")
parser.add_argument("--our_path", default="", type=str, help="VPT-deep results folder path")

parser.add_argument("--save_path", default="./result/source_target/save_result/", type=str, help="The folder path to save")

parser.add_argument("--rotate", default=True, help="Rotate a head direction")
parser.add_argument("--slice_x", type=int, help="")
parser.add_argument("--slice_y", type=int, help="")
parser.add_argument("--slice_z", type=int, help="")
parser.add_argument("--vmin", default=0, type=float, help="min value of error map histogram")
parser.add_argument("--vmax", default=1, type=float, help="max value of error map histogram")

def compute_MSE(img1, img2):
    return ((img1 - img2) ** 2).mean()

def rmse(img1, img2):
    if type(img1) == torch.Tensor:
        return torch.sqrt(compute_MSE(img1, img2)).item()
    else:
        return np.sqrt(compute_MSE(img1, img2))

def PSNR(recon, golden):
    if torch.is_tensor(recon):
        recon = np.array(recon.cpu(), dtype=np.float32)
        golden = np.array(golden.cpu(), dtype=np.float32)
    MSE = np.square(np.subtract(recon, golden)).mean()
    return 10 * np.log10(np.square(np.max(golden)) / MSE).item()


def make_dict(args):
    # Get patient numbers from json file
    with open (args.data_json, "r") as f:
        test_dict = json.load(f)["validation"]

    # Get file names from folder
    image_list = glob.glob(os.path.join(args.image_path, "*.nii"))
    label_list = glob.glob(os.path.join(args.label_path, "*.nii"))
    base_list = glob.glob(os.path.join(args.base_path, "*/*.nii"))
    fft_list = glob.glob(os.path.join(args.fft_path, "*/*.nii"))
    ln_list = glob.glob(os.path.join(args.ln_path, "*/*.nii"))
    bias_list = glob.glob(os.path.join(args.bias_path, "*/*.nii"))
    lora_list = glob.glob(os.path.join(args.lora_path, "*/*.nii"))
    adapter_list = glob.glob(os.path.join(args.adapter_path, "*/*.nii"))
    ssf_list = glob.glob(os.path.join(args.ssf_path, "*/*.nii"))
    shallow_list = glob.glob(os.path.join(args.shallow_path, "*/*.nii"))
    our_list = glob.glob(os.path.join(args.our_path, "*/*.nii"))
    
    # Make a dictionary of paths according to patient number
    path_dict = {}
    for data in test_dict:
        number = data["image"].split("_")[3]

        image = [image for image in image_list if image.split("/")[-1].split("_")[3]==number][0]
        label = [label for label in label_list if label.split("/")[-1].split("_")[3]==number][0]
        base = [base for base in base_list if base.split("/")[-1].split("_")[3]==number][0]
        fft = [fft for fft in fft_list if fft.split("/")[-1].split("_")[3]==number][0]
        ln = [ln for ln in ln_list if ln.split("/")[-1].split("_")[3]==number][0]
        bias = [bias for bias in bias_list if bias.split("/")[-1].split("_")[3]==number][0]
        lora = [lora for lora in lora_list if lora.split("/")[-1].split("_")[3]==number][0]
        adapter = [adapter for adapter in adapter_list if adapter.split("/")[-1].split("_")[3]==number][0]
        ssf = [ssf for ssf in ssf_list if ssf.split("/")[-1].split("_")[3]==number][0]
        shallow = [shallow for shallow in shallow_list if shallow.split("/")[-1].split("_")[3]==number][0]
        our = [our for our in our_list if our.split("/")[-1].split("_")[3]==number][0]
  
    
        path_dict[number] = {
            "label": label,
            "image": image,
            "base": base,
            "fft": fft,
            "ln": ln,
            "bias": bias,
            "lora": lora,
            "adapter": adapter,
            "ssf": ssf,
            "shallow": shallow,
            "our": our
            }

    return path_dict


def load_and_transform(path):
    # Load nii file and transform to array type
    image = nib.load(path["image"]).get_fdata().squeeze(-1)
    label = nib.load(path["label"]).get_fdata().squeeze(-1)
    base = nib.load(path["base"]).get_fdata()
    fft = nib.load(path["fft"]).get_fdata()
    ln = nib.load(path["ln"]).get_fdata()
    bias = nib.load(path["bias"]).get_fdata()
    lora = nib.load(path["lora"]).get_fdata()
    adapter = nib.load(path["adapter"]).get_fdata()
    ssf = nib.load(path["ssf"]).get_fdata()
    shallow = nib.load(path["shallow"]).get_fdata()
    our = nib.load(path["our"]).get_fdata()
    
    # Normalize
    Intensity = transforms.ScaleIntensity(minv=0.0, maxv=1.0)
    image = Intensity(image)
    label = Intensity(label)
    base = torch.from_numpy(base)
    fft = torch.from_numpy(fft)
    ln = torch.from_numpy(ln)
    bias = torch.from_numpy(bias)
    lora = torch.from_numpy(lora)
    adapter = torch.from_numpy(adapter)
    ssf = torch.from_numpy(ssf)
    shallow = torch.from_numpy(shallow)
    our = torch.from_numpy(our)
    
    # Rotate brain directions if needed
    if args.rotate:
        image = rearrange(image, 'h w d -> w h d')
        label = rearrange(label, 'h w d -> w h d')
        base = rearrange(base, 'h w d -> w h d')
        fft = rearrange(fft, 'h w d -> w h d')
        bias = rearrange(bias, 'h w d -> w h d')
        ln = rearrange(ln, 'h w d -> w h d')
        lora = rearrange(lora, 'h w d -> w h d')
        adapter = rearrange(adapter, 'h w d -> w h d')
        ssf = rearrange(ssf, 'h w d -> w h d')
        shallow = rearrange(shallow, 'h w d -> w h d')
        our = rearrange(our, 'h w d -> w h d')

    fdata = {
        "label": label,
        "image": image,
        "base": base,
        "fft": fft,
        "ln": ln,
        "bias": bias,
        "lora": lora,
        "adapter": adapter,
        "ssf": ssf,
        "shallow": shallow,
        "our": our
        }

    
    return fdata


def compute_performance(image, label):
    image = image.unsqueeze(0).unsqueeze(0)
    label = label.unsqueeze(0).unsqueeze(0)

    p = PSNR(image, label)
    data_range=label.max().unsqueeze(0)

    s = SSIMMetric(data_range=data_range,spatial_dims=3)._compute_metric(image, label)

    r = rmse(image, label)
    nrmse = r / (label.max()-label.min())

    return np.mean(p), np.mean(s), np.mean(nrmse)


def compute_diff(fdata):
    # image = [errormap, performance[p, s, r]]
    image = [abs(fdata["label"]-fdata["image"]), compute_performance(fdata["image"], fdata["label"])]
    base = [abs(fdata["label"]-fdata["base"]), compute_performance(fdata["base"], fdata["label"])]
    fft = [abs(fdata["label"]-fdata["fft"]), compute_performance(fdata["fft"], fdata["label"])]
   ln = [abs(fdata["label"]-fdata["ln"]), compute_performance(fdata["ln"], fdata["label"])]
    bias = [abs(fdata["label"]-fdata["bias"]), compute_performance(fdata["bias"], fdata["label"])]
    lora = [abs(fdata["label"]-fdata["lora"]), compute_performance(fdata["lora"], fdata["label"])]
    adapter = [abs(fdata["label"]-fdata["adapter"]), compute_performance(fdata["adapter"], fdata["label"])]
    ssf = [abs(fdata["label"]-fdata["ssf"]), compute_performance(fdata["ssf"], fdata["label"])]
    shallow = [abs(fdata["label"]-fdata["shallow"]), compute_performance(fdata["shallow"], fdata["label"])]
    our = [abs(fdata["label"]-fdata["our"]), compute_performance(fdata["our"], fdata["label"])]

    error = {
            "image": image,
            "base": base,
            "fft": fft,
            "ln": ln,
            "bias": bias,
            "lora": lora,
            "adapter": adapter,
            "ssf": ssf,
            "shallow": shallow,
            "our": our
            }

    

    return error

def slice_output(image, img_name, save_path, cmap='gray') :
    if args.slice_z is None:
        args.slice_z = int(image.shape[2]/2)
    slice=image[:,:,args.slice_z]
    plt.imshow(slice, origin="lower", cmap=cmap)
    plt.axis('off')
    plt.savefig(f'{save_path}/{img_name}.png', bbox_inches='tight', pad_inches=0)  

def main(args):
    path_dict = make_dict(args)
    number_list = list(path_dict.keys())

    for number in number_list:
        print(f"Plot {number}")

        fdata = load_and_transform(path_dict[number])
        error = compute_diff(fdata)
        name = list(fdata.keys())

        save_path = os.path.join(args.save_path, number)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(len(fdata)):
            idx = name[i]
            slice_output(fdata[name[i]], f"{idx}", save_path)
            if idx != 'label':
                slice_output(error[idx][0], f"{idx}: {error[idx][1]}", save_path, cmap='BuPu')
        print("save!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)