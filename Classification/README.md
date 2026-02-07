# Brain Tumor Multi-Class Classification with ResNet50

A comprehensive deep learning project for multi-class classification of brain tumors using MRI scans with ResNet50 architecture and advanced data processing techniques.

## Overview

This project implements a sophisticated brain tumor classification system using transfer learning with ResNet50. The model is trained to classify MRI brain scans into four distinct categories: glioma, meningioma, pituitary tumors, and no tumor, achieving high accuracy through advanced preprocessing and multi-GPU training strategies.

## Key Features

- **Multi-Class Classification**: 4-class tumor type classification (glioma, meningioma, pituitary, no tumor)
- **Multi-Dataset Integration**: Merges 11 different brain MRI datasets for comprehensive training
- **Multi-GPU Training**: Utilizes TensorFlow's MirroredStrategy for distributed training across multiple GPUs
- **Advanced Data Augmentation**: Random flips and brightness variations for enhanced model robustness
- **Transfer Learning**: ResNet50 with frozen base layers and custom classification head
- **Memory-Efficient Processing**: Streaming data pipelines with prefetching for optimal performance

## Dataset

### Data Sources

The model integrates data from 11 different brain MRI datasets, providing comprehensive coverage and generalization:

- **Brain MRI Scans for Brain Tumor Classification**
- **Brain Tumor MRI Scans**
- **Brain Tumor MRI Dataset**
- **Brain Tumor MRI Dataset for Deep Learning**
- **Brain Tumor Classification**
- **Brain Tumor Classification MRI**
- **Brain Tumors Dataset**

### Dataset Statistics

After merging all datasets, the final distribution is:

- **Total Images**: 46,920 MRI scans
- **Training Set**: 31,651 images (70%)
- **Validation Set**: 7,631 images (15%)
- **Test Set**: 7,638 images (15%)

### Class Distribution

- **Glioma**: 14,994 images
- **Meningioma**: 15,454 images
- **Pituitary**: 13,084 images
- **No Tumor**: 9,282 images

## Technical Implementation

### Architecture

The model leverages ResNet50 as the base architecture, pre-trained on ImageNet, with a custom classification head:

- **Base Model**: ResNet50 (pre-trained on ImageNet, frozen)
- **Custom Head**:
  - Global Average Pooling layer for spatial feature aggregation
  - Dropout(0.5) for regularization and overfitting prevention
  - Dense(4) layer with Softmax activation for multi-class classification

### Training Strategy

The training approach incorporates several advanced techniques:

- **Learning Rate**: 1e-4 with Adam optimizer
- **Batch Size**: 128 samples per batch for efficient GPU utilization
- **Epochs**: 10 epochs with comprehensive monitoring
- **Loss Function**: Categorical Crossentropy for multi-class classification
- **Optimizer**: Adam optimizer for efficient gradient-based optimization
- **Metrics**: Accuracy for performance evaluation

### Data Preprocessing Pipeline

A sophisticated preprocessing pipeline ensures optimal input quality:

1. **Image Resizing**: All images resized to 224x224 pixels for ResNet50 compatibility
2. **ResNet Preprocessing**: Applied using `resnet50.preprocess_input()` for optimal model input
3. **Data Augmentation** (Training only):
   - Random horizontal flips for increased dataset diversity
   - Random brightness variations (±10%) to simulate different imaging conditions
4. **Dataset Pipeline**: TensorFlow data pipelines with prefetching for optimal performance

## Performance

### Training Results

The model achieved excellent performance across all metrics:

- **Final Training Accuracy**: 85.52%
- **Final Validation Accuracy**: 87.33%
- **Test Accuracy**: 87.41%
- **Test Loss**: 0.3582

### Training Progress

The model demonstrated consistent improvement throughout training:

- **Epoch 1**: Training Acc: 39.40%, Val Acc: 73.61%
- **Epoch 5**: Training Acc: 77.77%, Val Acc: 84.06%
- **Epoch 10**: Training Acc: 85.52%, Val Acc: 87.33%

### Training Efficiency

- **Multi-GPU Setup**: Utilized 2 GPUs (Tesla T4) with MirroredStrategy
- **Training Time**: ~125 seconds per epoch
- **Batch Processing**: 248 batches per epoch with 128 samples each

## Model Architecture Details

### Transfer Learning Strategy

The transfer learning approach focuses on feature reuse:

- **Frozen Base Model**: Entire ResNet50 base model remains frozen to preserve pre-trained features
- **Trainable Head**: Only the custom classification layers are trained
- **Rationale**: This strategy leverages powerful pre-trained visual features while adapting to medical imaging characteristics

### Multi-GPU Optimization

The training process incorporates sophisticated distributed training:

- **MirroredStrategy**: Synchronous training across multiple GPUs
- **Data Parallelism**: Each GPU processes a subset of the batch
- **Gradient Synchronization**: Gradients are averaged across GPUs for consistent updates

## Key Insights

### Multi-Dataset Benefits

Training on multiple diverse datasets significantly improved the model's ability to handle variations in:

- Imaging equipment and protocols
- Patient populations and demographics
- Tumor presentation and characteristics
- Image quality and resolution

### Transfer Learning Effectiveness

The frozen ResNet50 base proved highly effective for medical imaging:

- Rapid convergence within 10 epochs
- High baseline performance from first epoch
- Stable training with minimal overfitting

### Data Augmentation Impact

Simple augmentation techniques provided significant benefits:

- Random flips increased model robustness to orientation variations
- Brightness variations improved handling of different imaging conditions

## Applications

This brain tumor classification system has potential applications in:

- **Clinical Diagnosis**: Automated tumor type identification for radiologists
- **Medical Screening**: Preliminary assessment and triage of MRI scans
- **Research Tool**: Large-scale analysis of brain tumor imaging datasets
- **Educational Purpose**: Demonstrating multi-class medical image classification
- **Treatment Planning**: Supporting tumor type-specific treatment decisions

## Performance Metrics

The model's performance was evaluated using comprehensive metrics:

### Overall Performance

- **Test Accuracy**: 87.41%
- **Test Loss**: 0.3582
- **Validation Accuracy**: 87.33%

### Training Progression

- **Convergence**: Steady improvement over 10 epochs
- **Overfitting**: Minimal gap between training and validation accuracy
- **Stability**: Consistent performance across epochs

## Future Improvements

- **Fine-Tuning**: Unfreeze some ResNet50 layers for domain-specific adaptation
- **Advanced Augmentation**: Implement more sophisticated medical imaging augmentations
- **Ensemble Methods**: Combine multiple architectures for improved accuracy
- **Explainability**: Add Grad-CAM visualizations for model interpretability
- **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
- **Hyperparameter Optimization**: Systematic hyperparameter tuning for better performance

## Class Details

### Glioma Tumors

- **Description**: Tumors arising from glial cells in the brain
- **Characteristics**: Infiltrative growth patterns, variable appearance on MRI
- **Clinical Significance**: Most common primary brain tumor type

### Meningioma Tumors

- **Description**: Tumors arising from meninges (brain covering)
- **Characteristics**: Well-circumscribed, extra-axial location
- **Clinical Significance**: Generally benign but can cause compression symptoms

### Pituitary Tumors

- **Description**: Tumors arising from pituitary gland
- **Characteristics**: Sellar region location, hormonal effects
- **Clinical Significance**: Can affect endocrine function

### No Tumor

- **Description**: Normal brain MRI scans
- **Characteristics**: Normal anatomical structures, no pathological findings
- **Clinical Significance**: Baseline for comparison and healthy controls
