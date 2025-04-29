# BraTS Segmentation with MONAI

A deep learning-based brain tumor segmentation service using MONAI framework and SegResNet model. This project provides both a FastAPI web service and a Jupyter notebook implementation for brain tumor segmentation using the BraTS dataset.

## Features

- Brain tumor segmentation using SegResNet architecture
- FastAPI-based REST API service
- Docker containerization support
- MONAI-based data preprocessing and augmentation
- Support for multiple tumor regions:
  - TC (Tumor Core)
  - WT (Whole Tumor)
  - ET (Enhancing Tumor)

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Docker (optional)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd monai_brats_segmentation

2. Install dependencies:
```bash
pip install -r requirements.txt

## Usage

### Running the FastAPI Service
```bash
python monai_brats_segmentation.py

The service will be available at http://localhost:8015

API Endpoints:
GET /: Health check endpoint
GET /model-status: Check model loading status
POST /predict: Submit MRI scans for tumor segmentation

### Using docker

1. Build the Docker image:
```bash
docker build -t brats-segmentation .

2. Run the container:
```bash
docker run -p 8015:8015 --gpus all brats-segmentation

# Testing

Use the provided test.py script to test the API:
```bash
python test.py

# Jupyter Notebook
The monai_brats_segmentation.ipynb notebook provides a step-by-step implementation including:

- Environment setup
- Data preprocessing
- Model architecture
- Training configuration
- Inference examples
- Visualization tools
