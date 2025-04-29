# BraTS Segmentation with MONAI

A deep learning-based brain tumor segmentation service using MONAI framework and SegResNet model. Provides both FastAPI web service and Jupyter notebook implementations for brain tumor segmentation using the BraTS dataset.

## Features

- **Segmentation Architecture**: SegResNet with MONAI integration
- **Deployment Options**:
  - FastAPI REST API service
  - Jupyter Notebook interactive implementation
- **Multi-region Tumor Segmentation**:
  - TC (Tumor Core)
  - WT (Whole Tumor)
  - ET (Enhancing Tumor)
- **Efficient Processing**:
  - MONAI-based data preprocessing
  - GPU-accelerated augmentation
- **Containerization**: Docker support with GPU compatibility

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Docker Engine 20.10+ (for containerized deployment)
- NVIDIA Container Toolkit (for GPU support in Docker)

## Installation

1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/monai-brats-segmentation.git
   cd monai_brats_segmentation
