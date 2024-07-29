<div align="center">
    <h1>Parameter Efficient Fine Tuning for <br> Multi-scanner PET to PET Reconstruction <br> MICCAI 2024</h1>
</div>

<div align="center">
    <h3>Yumin Kim*, Gayoon Choi*, Seong Jae Hwang <br> Yonsei University <br> (* Equal Contributor) 
</h3>
                                            
</div>


<div align="center">
  <h3>
    <a href="https://arxiv.org/pdf/2407.07517">Paper</a> |
    <a href="http://MICV-yonsei.github.io/petite2024/">Project Page</a>
  </h3>
</div>

 ![STR](https://github.com/mineeuk/PETITE/assets/72694034/2641a7f3-facc-4eac-84cf-b96ea3c32f64)
 
**Accepted @ MICCAI 2024** \
We will release the code soon ! 🦍

### Requirements
- [3D CVT-GAN](https://github.com/Aru321/GAN-for-PET-Reconstruction)
```
conda env create -f cvt.yaml
conda activate cvt
```
- [UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV)
```
conda env create -f unetr.yaml
conda activate unetr
```

### Dataset Preparation

For datasets from [ADNI(Alzheimer's Disease Neuroimaging Initiative)](https://ida.loni.usc.edu/login.jsp?project=ADNI) that include PET(Positron emission tomography) scans.
```
Data Collection/full_FDG/
```
Your dataset directory should be structured as follows:
- training set: /root_dir/ADNI/Dynamic/Resolution/
- validation set: /root_dir/ADNI/Averaged/Resolution/

| Scanner | Resolution       | Voxel spacing             | Manufacturer        | Institution        |
|---------|------------------|---------------------------|---------------------|--------------------|
| 1       | (192, 192, 136)  | (1.21875, 1.21875, 1.21875)| Siemens             | Univ of California |
| 2       | (192, 192, 128)  | (1.21875, 1.21875, 1.21875)| Siemens             | Univ of California |
| 3       | (224, 224, 81)   | (1.01821, 1.01821, 2.02699)| Siemens             | Univ of California |
| 4       | (128, 128, 90)   | (2, 2, 2)                  | Philips Healthcare  | OHSU               |
| 5       | (128, 128, 63)   | (2.05941, 2.05941, 2.425)  | Siemens             | UCSD               |

To create an json file for efficient data split, run the following command:
```commandline
sh data/make_json.sh
```

### Downloading pre-trained weights
Click the links below to download the pre-trained weights for each of the five scanners. Each scanner has weights for three folds. 
Training details are described in our paper. Currently, available versions of pre-trained weights are as follows:
- [Scanner 1-5](https://drive.google.com/drive/folders/1nk-6n3KY8XGIn2TMxDRNZ5awdz5FYGnM)

### Pre-training
```commandline
sh shell/train/pretraining.sh
```

### PEFT
```commandline
sh shell/train/tuning.sh
```

### How to Apply LoRA in Conv3D
Using Low-Rank Adaptation (LoRA) with Conv3D involves modifying the Conv3D layers to integrate the LoRA technique.

- Modified to place Lora instead of Convolutional Projection in Attention
- nn.Conv3d -> lora.Conv3d
```
import loralib as lora
```

### ADNI policy
ADNI data cannot be publicly shared by private users. It was not possible to directly upload data in this repository. However, feel free to [contact me](yumin@yonsei.ac.kr) if you need more information about how to access data or about data used.

Remember to follow the ADNI data sharing and publication policy, if you are going to publish a work in which data from the ADNI database were used.