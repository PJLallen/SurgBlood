<h1 align="center">Synergistic Bleeding Region and Point Detection in <br>Laparoscopic Surgical Videos</h1>

<p align="center"><strong>CVPR 2026</strong></p>

<div align="center">
    <a href='https://scholar.google.com/citations?user=1lPivLsAAAAJ&hl=en' target='_blank'><strong>Jialun Pei</strong></a><sup>1</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=lvx5k9cAAAAJ' target='_blank'><strong>Zhangjun Zhou</strong></a><sup>2</sup>,&thinsp;
    <a href='https://scholar.google.com.hk/citations?user=yXycwhIAAAAJ&hl=zh-CN&oi=sra' target='_blank'><strong>Diandian Guo</strong></a><sup>1</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=1lPivLsAAAAJ&hl=en' target='_blank'><strong>Zhixi Li</strong></a><sup>2,3</sup>,&thinsp;
    <a href='https://harry-qinjing.github.io/' target='_blank'><strong>Jing Qin</strong></a><sup>2</sup>,&thinsp;
    <a href='https://scholar.google.com/citations?user=Shy1gnMAAAAJ&hl=en' target='_blank'><strong>Bo Du</strong></a><sup>4*</sup>,&thinsp;
    <a href='https://scholar.google.com.hk/citations?user=OFdytjoAAAAJ&hl=zh-CN&oi=sra' target='_blank'><strong>Pheng-Ann Heng</strong></a><sup>1</sup>
</div>

<div align="center">
    <sup>1</sup> The Chinese University of Hong Kong &ensp;
    <sup>2</sup> The Hong Kong Polytechnic University
    <br />
    <sup>3</sup> Southern Medical University &ensp;
    <sup>4</sup> Wuhan University
</div>

<br />

<div align="center">
  <a href='https://arxiv.org/abs/2503.22174'><img src='https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white' alt='arXiv Paper'></a>
  <a href='https://youtu.be/wueRsI2lZjU'><img src='https://img.shields.io/badge/Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white' alt='YouTube Demo'></a>
  <a href='https://drive.google.com/file/d/1Xw3Px0w2KVKY6IzzdY6fRjjHRLiI5is-/view?usp=sharing'><img src='https://img.shields.io/badge/Weights-Google_Drive-0F9D58?style=for-the-badge&logo=googledrive&logoColor=white' alt='Pre-trained Weights'></a>
  <a href='LICENSE'><img src='https://img.shields.io/badge/License-MIT-F2C94C?style=for-the-badge' alt='MIT License'></a>
</div>

<p align="center">
  A unified framework for <strong>bleeding region segmentation</strong> and <strong>bleeding point localization</strong> in laparoscopic surgical videos, together with the <strong>SurgBlood</strong> benchmark.
</p>

<p align="center">
  <a href="#highlights"><strong>Highlights</strong></a> •
  <a href="#supplementary-demo-video"><strong>Demo</strong></a> •
  <a href="#dataset-preparation"><strong>Dataset</strong></a> •
  <a href="#pre-trained-models"><strong>Weights</strong></a> •
  <a href="#usage"><strong>Usage</strong></a> •
  <a href="#citation"><strong>Citation</strong></a>
</p>

<div align="center">
  <img src="assets/Pipeline_v4_page-0001.jpg" width="760" alt="BlooDet pipeline overview" />
</div>

> This repository contains the official implementation of [**Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos**](https://arxiv.org/abs/2503.22174).

## Highlights

- Introduces **BlooDet**, a synergistic framework for joint bleeding region detection and bleeding point localization.
- Builds **SurgBlood**, a dedicated benchmark for laparoscopic bleeding analysis.
- Supports both **training** and **evaluation** for surgical video understanding workflows.
- Provides a **supplementary demo**, **pre-trained checkpoints**, and **visualization results** for quick exploration.

## Supplementary Demo Video

<div align="center">
  <a href="https://youtu.be/wueRsI2lZjU">
    <img src="assets/bloodet-demo-preview.gif" width="640" alt="BlooDet supplementary demo preview" />
  </a>
</div>

<div align="center">
  <a href="https://youtu.be/wueRsI2lZjU"><strong>Watch the full BlooDet supplementary demo on YouTube</strong></a>
</div>

<div align="center">
  Click the GIF preview above to open the full demo video on YouTube.
</div>

## Project Resources

- Paper: [arXiv](https://arxiv.org/abs/2503.22174)
- Demo: [YouTube supplementary video](https://youtu.be/wueRsI2lZjU)
- Pre-trained weights: [Google Drive](https://drive.google.com/file/d/1Xw3Px0w2KVKY6IzzdY6fRjjHRLiI5is-/view?usp=sharing)
- Visualization results: [Google Drive](https://drive.google.com/file/d/1XrC6q8BftPLTIq8gLe7YT0uyQbWqRmxI/view?usp=sharing)
- Contact: `peijialun@gmail.com`

## Environment Preparation

### Requirements

- Please refer to [SAM2](https://github.com/facebookresearch/sam2) for the base environment setup.
- You may need to install `Apex` with `pip` depending on your environment.

## Dataset Preparation

### Download the datasets and annotation files

- **SurgBlood**: coming by June 2026.

### Register datasets

1. Download the datasets and place them in the same root folder. To match the folder names used in the dataset mappers, it is recommended not to rename them. The structure should look like:

```text
DATASET_ROOT/
├── SurgBlood
│   ├── train
│   │   ├── videos-image
│   │   ├── videos-mask
│   │   ├── videos-mask-edge
│   │   └── videos-point
│   └── test
│       ├── videos-image
│       ├── videos-mask
│       └── videos-point
```

2. For convenience, we provide a test dataset folder containing four types of bleeding:

<div align="center">
  <img src="assets/Supp_Data_V1_page-0001.jpg" width="1000" alt="Dataset example" />
</div>

## Pre-trained Models

- Download the pre-training weights of `sam2_base`: [sam2_hiera_base_plus](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)
- Download the pre-trained weights on SurgBlood: [Google Drive](https://drive.google.com/file/d/1Xw3Px0w2KVKY6IzzdY6fRjjHRLiI5is-/view?usp=sharing)

## Visualization Results

The visualization results of **state-of-the-art methods** on the **SurgBlood test set** are available on [Google Drive](https://drive.google.com/file/d/1XrC6q8BftPLTIq8gLe7YT0uyQbWqRmxI/view?usp=sharing).

## Usage

### Train & Test

- To train and evaluate BlooDet on a single GPU, run:

```shell
bash trainAndEvaluate.sh
```

- Alternatively, to test or evaluate BlooDet on SurgBlood, run:

```shell
python test.py
bash evaluate.sh
```

## Citation

If this repository is useful for your research, please consider citing:

```bibtex
@misc{pei2025synergisticbleedingregionpoint,
  title={Synergistic Bleeding Region and Point Detection in Laparoscopic Surgical Videos},
  author={Jialun Pei and Zhangjun Zhou and Diandian Guo and Zhixi Li and Jing Qin and Bo Du and Pheng-Ann Heng},
  year={2025},
  eprint={2503.22174},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.22174}
}
```
