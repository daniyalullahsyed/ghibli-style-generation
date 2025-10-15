# ghibli-style-generation
Comparing GAN and Diffusion models to transform photos into the Studio Ghibli art style. Includes the official code and a new open-source dataset for style transfer.

Ghibli-Style Image Generation
This repository contains the code and resources for the research project "Ghibli-Style Image Generation from Scratch and Pretrained Models", which explores deep generative models for stylizing real-world photographs into the aesthetic of Studio Ghibli.

Abstract
This project presents a dual-approach investigation into stylizing real-world photographs into the whimsical aesthetic of Studio Ghibli using deep generative models under low-data constraints. Due to the lack of publicly available paired datasets, we curated and augmented a novel collection of 400 image pairs.

Group 1 built a Pix2PixHD-inspired GAN architecture from scratch, training a ResNet-based generator with adversarial, feature-matching, and perceptual losses.

Group 2 fine-tuned a pretrained Stable Diffusion v1.5 model with Low-Rank Adaptation (LoRA) in a supervised image-to-image pipeline.

Our contributions include a reproducible GAN implementation, an efficient diffusion-based fine-tuning pipeline, and a new open-source dataset.

Repository Structure
ghibli-style-image-generation/
├── data/               # Instructions on how to download the dataset
├── notebooks/          # Jupyter Notebooks for experimentation
├── src/                # Python source code
├── models/             # Information on trained models
├── results/            # Generated images and evaluation results
├── .gitignore          # Files to be ignored by Git
├── LICENSE             # Project license
└── README.md           # This file
Dataset
We created a new open-source dataset named ghibli-illustration-dataset, which is available on Kaggle and Hugging Face. This dataset consists of 400 image pairs, each containing an original real-world photograph and its Ghibli-style illustrated version.

Models
This project explores two different models for Ghibli-style image generation:

Group 1: Pix2PixHD from Scratch
Authors: Sujay Pookkattuparambil, Daniyal Syed

Description: Implementation of a Pix2PixHD-inspired model from scratch.

Group 2: Fine-tuned Stable Diffusion
Authors: Amarsaikhan Batjargal, Talha Azaz

Description: Fine-tuning of a pretrained Stable Diffusion v1.5 model using LoRA.

Installation
To set up the environment and install the required dependencies, please follow these steps:

Clone the repository:

Bash

git clone https://github.com/your-username/ghibli-style-image-generation.git
cd ghibli-style-image-generation
Install the required packages:

Bash

pip install -r requirements.txt
Usage
You can use the provided Jupyter Notebook (CSC594_Project_GhibliGenerator.ipynb) located in the notebooks/ directory to explore the data, train the models, and generate images. The notebook is organized into sections for each group's approach.

Results
Our models were evaluated using FID, SSIM, and LPIPS metrics. The fine-tuned Stable Diffusion model achieved a final training loss of 0.0796 (MSE) with validation metrics of SSIM=0.71 and LPIPS=0.19, indicating decent structural and perceptual fidelity.

Here are some examples of the generated images:



Citation
If you use this work in your research, please cite our paper:

@article{ghibli_style_2025,
  title={Ghibli-Style Image Generation from Scratch and Pretrained Models},
  author={Batjargal, Amarsaikhan and Pookkattuparambil, Sujay and Azaz, Talha and Syed, Daniyal},
  year={2025},
  journal={arXiv preprint arXiv:XXXX.XXXXX}
}
License
This project is licensed under the MIT License. See the LICENSE file for details.
