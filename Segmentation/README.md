# Brain Tumour Segmentation using nnU-Net

This project implements a 3D U-Net architecture for automated brain tumour segmentation using the BraTS2020 dataset. The implementation follows the nnU-Net methodology, which provides a robust framework for medical image segmentation tasks.

## Overview

The project focuses on segmenting brain tumours from multimodal MRI scans. It utilizes four different MRI modalities (FLAIR, T1, T1CE, T2) to identify and classify three tumour regions:

- **Whole Tumour (WT)**: All tumour regions combined
- **Tumour Core (TC)**: Enhancing tumour and necrotic core
- **Enhancing Tumour (ET)**: Active tumour regions

## Dataset

The implementation is trained on the BraTS2020 dataset, which contains:

- 369 training cases with multimodal MRI scans
- Four MRI modalities per case: FLAIR, T1, T1CE, and T2
- Manual segmentation annotations provided by expert radiologists
- Standardized preprocessing and quality control

## Architecture

The model employs a 3D U-Net architecture specifically designed for volumetric medical image segmentation:

### Network Structure

- **Encoder Path**: Four downsampling levels with feature maps [32, 64, 128, 256]
- **Bottleneck**: Deep feature extraction at the lowest resolution
- **Decoder Path**: Symmetric upsampling with skip connections
- **Multi-scale Outputs**: Deep supervision at three different resolutions

### Key Features

- 3D convolutional operations for volumetric context
- Instance normalization for stable training
- LeakyReLU activations for gradient flow
- Skip connections to preserve spatial information
- Deep supervision for improved gradient propagation

## Training Methodology

### Data Processing

- **Patch-based Training**: Random 128×128×128 patches extracted from full volumes
- **Intensity Normalization**: Z-score normalization within brain regions
- **Data Augmentation**: Random flips along three axes
- **Balanced Sampling**: 66% of patches centered on tumour regions

### Loss Function

The model uses a hybrid loss combining:

- **Dice Loss**: Optimizes overlap between predicted and ground truth segmentation
- **Binary Cross-Entropy**: Provides pixel-wise classification guidance
- **Multi-scale Loss**: Deep supervision at different resolutions

### Training Strategy

- **Optimizer**: SGD with Nesterov momentum (0.99)
- **Learning Rate Schedule**: Polynomial decay with power 0.9
- **Mixed Precision Training**: Automatic mixed precision for memory efficiency
- **Batch Size**: 12 patches per batch
- **Training Duration**: 2 epochs with comprehensive validation

## Performance Metrics

The model achieves competitive segmentation performance measured by Dice scores:

### Validation Results

- **Whole Tumour (WT)**: 0.847 Dice score
- **Tumour Core (TC)**: 0.753 Dice score
- **Enhancing Tumour (ET)**: 0.761 Dice score

### Training Characteristics

- **Convergence**: Stable loss reduction across epochs
- **Efficiency**: ~700 seconds per epoch with GPU acceleration
- **Memory Optimization**: 10.2 MB model size with mixed precision

## Technical Implementation

### Data Pipeline

- **Efficient Loading**: Custom Dataset class with on-the-fly preprocessing
- **Memory Management**: Patch-based approach to handle large 3D volumes
- **Parallel Processing**: Multi-worker data loading for improved throughput

### Model Optimization

- **GPU Acceleration**: CUDA-enabled training with automatic mixed precision
- **Data Parallelism**: Multi-GPU support through nn.DataParallel
- **Gradient Scaling**: Prevents underflow in mixed precision training

## Clinical Relevance

This segmentation approach provides:

- **Automated Analysis**: Reduces manual annotation time for radiologists
- **Quantitative Metrics**: Objective tumour volume measurements
- **Treatment Planning**: Precise tumour delineation for radiotherapy
- **Progress Monitoring**: Consistent segmentation for longitudinal studies

## Model Architecture Details

### Convolutional Blocks

Each block consists of:

- 3D convolution with 3×3×3 kernels
- Instance normalization for stable training
- LeakyReLU activation (0.01 negative slope)
- No bias terms for regularization

### Skip Connections

- Direct feature concatenation between encoder and decoder
- Preserves fine-grained spatial information
- Enables precise boundary localization

### Deep Supervision

- Auxiliary outputs at intermediate decoder levels
- Weighted loss combination (1.0, 0.5, 0.25)
- Improves gradient flow and convergence

## Data Characteristics

### Input Modalities

- **FLAIR**: Fluid-attenuated inversion recovery for edema detection
- **T1**: T1-weighted for anatomical structure
- **T1CE**: T1-weighted with contrast for enhancing regions
- **T2**: T2-weighted for tissue characterization

### Preprocessing Pipeline

- Brain extraction and skull stripping
- Intensity normalization within brain mask
- Standardized orientation and spacing
- Quality control for missing modalities

## Training Efficiency

### Computational Requirements

- **Memory**: Optimized for GPU memory constraints
- **Speed**: ~8-10 seconds per batch iteration
- **Scalability**: Supports multi-GPU training
- **Storage**: Efficient model serialization

### Optimization Techniques

- Mixed precision training for memory efficiency
- Gradient accumulation for large effective batch sizes
- Learning rate scheduling for stable convergence
- Early stopping based on validation metrics

## Segmentation Quality

The model demonstrates:

- **High Accuracy**: Competitive Dice scores across all tumour regions
- **Robust Performance**: Consistent results across different cases
- **Fine Detail**: Precise boundary localization
- **Clinical Utility**: Suitable for real-world medical applications

## Future Applications

This segmentation framework can be extended to:

- Other brain tumour types and datasets
- Different imaging modalities and protocols
- Real-time segmentation for clinical workflows
- Integration with treatment planning systems
- Longitudinal tumour progression analysis
