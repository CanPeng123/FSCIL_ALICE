# FSCIL_ALICE

This project hosts the code for implementing the ALICE algorithm for few-shot class-incremental classification, as presented in our paper:

## [Few-Shot Class-Incremental Learning from an Open-Set Perspective]

Can Peng, Kun Zhao, Tianren Wang, Meng Li, Brian C. Lovell; In: ECCV 2022.

[arXiv preprint](https://arxiv.org/abs/2208.00147).

# Training

**main_base.py**: training for the base task. 

**main_inc_ncm.py**: evaluation for the base and incremental tasks.

### base session

**run_base_CIFAR100.sh**: config and dataset settings for base session model trained on CIFAR100 dataset.

**run_base_CUB200.sh**: config and dataset settings for base session model trained on CUB200 dataset.

**run_base_miniImageNet.sh**: config and dataset settings for base session model trained on miniImageNet dataset.

### incremental session 

**run_inc_ncm_CIFAR100.sh**: config and dataset settings for incremental session evaluation on CIFAR100 dataset.

**run_inc_ncm_CUB200.sh**: config and dataset settings for incremental session evaluation on CUB200 dataset.

**run_inc_ncm_miniImageNet.sh**: config and dataset settings for incremental session evaluation on miniImageNet dataset.

To perform experiments on different few-shot settings (5-shot or 1-shot), please modify the input arguments - "used_img" and "balanced" accordingly.

## Acknowledgements
Our FSCIL ALICE implementation is based on [CEC-CVPR2021](https://github.com/icoz69/CEC-CVPR2021). We thank the authors for making their code public.

