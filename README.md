# Parameter Efficient Fine Tuning for Multi-scanner PET to PET Reconstruction

**Accepted @ MICCAI 2024** \
We will release the code soon ! 🦍

[[Project Page]](http://MICV-yonsei.github.io/petite2024/) [[arXiv]](http://MICV-yonsei.github.io/petite2024/)

![STR](https://github.com/mineeuk/PETITE/assets/72694034/2641a7f3-facc-4eac-84cf-b96ea3c32f64)


> **Parameter Efficient Fine Tuning for Multi-scanner PET to PET Reconstruction** \
> International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024 \
> Yumin Kim*, Gayoon Choi*, Seong Jae Hwang \
Yonsei University

### Abstract
Reducing scan time in Positron Emission Tomography (PET) imaging while maintaining high-quality images is crucial for minimizing patient discomfort and radiation exposure. Due to the limited size of datasets and distribution discrepancy across scanners in medical imaging, fine-tuning in a parameter-efficient and effective manner is on the rise. Motivated by the potential of Parameter-Efficient Fine-Tuning (PEFT), we aim to address these issues by effectively leveraging PEFT to improve limited data and GPU resource issues in multi-scanner setups. In this paper, we introduce **PETITE**, Parameter-Efficient Fine-Tuning for MultI-scanner PET to PET REconstruction that uses fewer than 1% of the parameters. To the best of our knowledge, this study is the first to systematically explore the efficacy of diverse PEFT techniques in medical imaging reconstruction tasks via prevalent encoder-decoder-type deep models. This investigation, in particular, brings intriguing insights into PETITE as we show further improvements by treating encoder and decoder separately and mixing different PEFT methods, namely, **Mix-PEFT**. Using multi-scanner PET datasets comprised of five different scanners, we extensively test the cross-scanner PET scan time reduction performances (i.e., a model pre-trained on one scanner is fine-tuned on a different scanner) of 21 feasible Mix-PEFT combinations to derive optimal PETITE. We show that training with less than 1% parameters using PETITE performs on par with full fine-tuning (i.e., 100% parameter).

## How to Apply LoRA in Conv3D
Using Low-Rank Adaptation (LoRA) with Conv3D involves modifying the Conv3D layers to integrate the LoRA technique.

### Citation
If you found this code useful, please cite the following paper:
```bibtex
@InProceedings{2024petite,
  author    = {Kim, Yumin and Choi, Gayoon and Hwang, Seong Jae},
  title     = {Parameter Efficient Fine Tuning for Multi-scanner PET to PET Reconstruction},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  month     = {June},
  year      = {2024}
}
