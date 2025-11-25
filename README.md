# RePiD: Recursive Pixel-based Diffusion Model

## Overview
<img width="2712" height="1530" alt="image" src="https://github.com/user-attachments/assets/140a4466-73f8-49c6-b33b-31a288e9d078" />
This repository contains the implementation of the Recursive Pixel-based Diffusion Model (RePiD), a cutting-edge approach to enhance image synthesis tasks. RePiD operates directly in the pixel space, employing a novel recursive mechanism and multiscale diffusion processes. Our model leverages hierarchical patch-based computations, significantly improving generation quality while reducing computational costs. 

## Research Context
Recent advancements in generative models have underscored the benefits of pixel-based methods over traditional latent-space approaches. Conventional diffusion models often suffer from high computation costs due to full-resolution operations. RePiD addresses these challenges by directly manipulating pixel space through recursive processing, allowing for enhanced efficiency and fidelity in tasks such as class-to-image generation and image-to-image translation.

<img width="2372" height="1652" alt="image" src="https://github.com/user-attachments/assets/09e3117c-212a-4150-b1fe-23e66573fc4c" />
## Contributions
- **End-to-End Recursive Framework**: RePiD introduces a recursive diffusion paradigm that efficiently generates images in pixel space, mitigating the computational burdens of high-resolution tasks.
- **Multiscale Processing**: By segmenting images into non-overlapping patches at multiple scales, RePiD performs efficient hierarchical computations.
- **Neighborhood Embedding**: Integrating a neighborhood encoding mechanism allows RePiD to maintain spatial coherence and capture complex relationships among image patches, enhancing the quality of generated images.
- **Open Source**: The complete code and models are available to promote reproducibility and further exploration in the field.

## Features
- High-quality image generation capabilities with minimal resource usage.
- Competitive performance evaluated against state-of-the-art models in class-to-image and image-to-image tasks.
- User-friendly implementation ready for experimentation and adaptation.

## Getting Started
To begin using RePiD, follow the installation instructions and usage guidelines below.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Recursive-Pixel-based-Diffusion-Model.git
   cd Recursive-Pixel-based-Diffusion-Model
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
To train the RePiD model, use the following command:
```bash
python scripts/train.py --config configs/train_config.yaml
```

For inference, run:
```bash
python scripts/inference.py --model path/to/trained/model --input path/to/input/image --output path/to/output/image
```
