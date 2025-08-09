# 🧠 Brain Tumor Segmentation with 3D U-Net

A deep learning project for automated brain tumor segmentation using 3D U-Net architecture on the BraTS2020 dataset. This project implements a complete pipeline from data preprocessing to model deployment with an interactive Streamlit web application.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Inference](#inference)
- [Web Application](#web-application)
- [Results](#results)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

Brain tumor segmentation is a critical task in medical image analysis that helps radiologists and clinicians in diagnosis, treatment planning, and monitoring. This project implements a 3D U-Net model to automatically segment brain tumors from multimodal MRI scans into four distinct regions:

- **Background** (Label 0)
- **Necrotic/Non-enhancing tumor core** (Label 1)
- **Peritumoral edema** (Label 2)
- **GD-enhancing tumor** (Label 3)

## ✨ Features

- **3D U-Net Architecture**: State-of-the-art deep learning model for volumetric segmentation
- **Multi-modal Input**: Processes FLAIR, T1, T1CE, and T2 MRI sequences simultaneously
- **Patch-based Training**: Efficient memory usage with overlapping patch extraction
- **Data Augmentation**: TorchIO-based augmentations for robust training
- **Combined Loss Function**: Dice Loss + Cross-Entropy for optimal performance
- **Interactive Web App**: User-friendly Streamlit interface for model deployment
- **Comprehensive Evaluation**: Dice scores and volume statistics for each tumor region
- **Early Stopping**: Prevents overfitting with patience-based training

## 📊 Dataset

This project uses the **BraTS2020 (Multimodal Brain Tumor Segmentation Challenge 2020)** dataset:

- **Training**: 369 cases with ground truth segmentations
- **Validation**: 125 cases (used for final evaluation)
- **Modalities**: T1, T1CE, T2, FLAIR (240×240×155 voxels each)
- **Annotations**: Expert-verified tumor segmentations

### Data Preprocessing

- **Normalization**: Z-score normalization per modality
- **Label Conversion**: Maps label 4 → 3 for consistency
- **Patch Extraction**: 128×128×64 patches with configurable stride
- **Tumor-aware Sampling**: Balanced sampling of tumor vs background patches

## 🏗️ Model Architecture

### 3D U-Net Implementation

The model follows the U-Net architecture adapted for 3D volumetric data:

```
Input: [Batch, 4, 128, 128, 64] (4 MRI modalities)
├── Encoder Path
│   ├── Conv3D Block 1: [4 → 16] + MaxPool3D
│   ├── Conv3D Block 2: [16 → 32] + MaxPool3D
│   ├── Conv3D Block 3: [32 → 64] + MaxPool3D
│   ├── Conv3D Block 4: [64 → 128] + MaxPool3D
│   └── Bottleneck: [128 → 256]
└── Decoder Path
    ├── UpConv + Skip 4: [256+128 → 128]
    ├── UpConv + Skip 3: [128+64 → 64]
    ├── UpConv + Skip 2: [64+32 → 32]
    ├── UpConv + Skip 1: [32+16 → 16]
    └── Final Conv: [16 → 4] (classes)
Output: [Batch, 4, 128, 128, 64] (segmentation probabilities)
```

### Key Components

- **Residual Connections**: Optional residual blocks in convolution layers
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout**: 3D dropout layers for regularization (20% rate)
- **Skip Connections**: Preserves fine-grained spatial information

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for training

### Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install nibabel scikit-image torchsummary torchio
pip install streamlit matplotlib plotly pandas numpy tqdm
pip install opencv-python pillow seaborn

# For Colab/Kaggle environments
pip install kaggle nilearn
```

### Dataset Setup

1. **Kaggle API Setup** (for automatic download):
   ```bash
   # Place your kaggle.json in ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Download BraTS2020**:
   ```python
   # Automatic download (in notebook)
   !kaggle datasets download -d awsaf49/brats20-dataset-training-validation
   !unzip brats20-dataset-training-validation.zip -d brats20_data
   ```

## 🚀 Usage

### Training

Run the complete training pipeline:

```python
# Configure paths and hyperparameters
TRAIN_DATASET_PATH = '/path/to/BraTS2020_TrainingData'
NUM_EPOCHS = 16
BATCH_SIZE = 2
LEARNING_RATE = 1e-5

# Initialize model and training
model = UNet3D_BraTS(in_channels=4, num_classes=4, feat_channels=[16, 32, 64, 128, 256])
criterion = CombinedLoss(num_classes=4, alpha=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Start training
python train.py
```

### Key Training Features

- **Patch-based Learning**: Extracts overlapping 3D patches for efficient training
- **Data Augmentation**: Random flips, noise injection using TorchIO
- **Mixed Training**: Balanced tumor and background patches (configurable ratio)
- **Checkpointing**: Saves model state every epoch with best model tracking
- **Early Stopping**: Stops training when validation performance plateaus

### Inference

```python
# Load trained model
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Run inference on new patient
segmentation, probabilities = predict_segmentation(model, device, patient_volume)
```

## 🌐 Web Application

Launch the interactive Streamlit web application:

```bash
streamlit run streamlit_app.py
```

<img width="1915" height="1021" alt="ui" src="https://github.com/user-attachments/assets/cde70494-2a0d-42c6-a9d4-bd27ed9580d7" />
<img width="1852" height="868" alt="ui result" src="https://github.com/user-attachments/assets/51d5a629-b4b0-4196-afb1-1738618609b4" />



### Web App Features

- **File Upload**: Drag-and-drop interface for NIfTI files
- **Automatic Model Loading**: Detects and loads checkpoint files
- **Real-time Processing**: Live progress tracking during inference
- **Interactive Visualization**: 
  - Slice-by-slice navigation
  - Multi-modal image display
  - Segmentation overlays
  - Grid and single-row view options
- **Statistical Analysis**: 
  - Volume calculations per tumor region
  - Slice-specific statistics
  - Percentage breakdowns
- **Export Options**: Download segmentation masks and reports

### Usage Instructions

1. **Upload Files**: Select all 4 MRI modalities (FLAIR, T1, T1CE, T2)
2. **Load Model**: Choose from available checkpoint files
3. **Configure**: Adjust patch size and stride parameters
4. **Process**: Click "Run Segmentation" to start inference
5. **Explore**: Navigate through slices and analyze results
6. **Download**: Export segmentation masks and statistics

## 📈 Results

### Model Performance

| Metric | Score |
|--------|--------|
| Mean Dice Score | 0.85+ |
| Necrotic Core Dice | 0.82+ |
| Edema Dice | 0.88+ |
| Enhancing Dice | 0.85+ |

<img width="2532" height="1214" alt="results colab" src="https://github.com/user-attachments/assets/68230e7f-7ac0-4c00-838d-b3244e40f7f7" />


### Training Insights

- **Convergence**: Model typically converges within 8-16 epochs
- **Memory Usage**: ~8GB GPU memory for batch size 2
- **Processing Time**: ~2-3 minutes per patient (inference)
- **Patch Strategy**: 50% overlap provides optimal results

## 📁 File Structure

```
brain-tumor-segmentation/
│
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── kaggle.json              # Kaggle API credentials
│
├── notebooks/
│   └── LAST_TIME.ipynb      # Complete training notebook
│
├── src/
│   ├── model.py             # 3D U-Net architecture
│   ├── dataset.py           # BraTS dataset loader
│   ├── loss.py              # Loss functions
│   ├── train.py             # Training pipeline
│   └── utils.py             # Utility functions
│
├── streamlit_app.py         # Web application
├── best_model.pth           # Trained model checkpoint
│
├── data/
│   └── brats20_data/        # Dataset directory
│       ├── BraTS2020_TrainingData/
│       └── BraTS2020_ValidationData/
│
└── results/
    ├── training_plots.png   # Loss and dice curves
    ├── sample_predictions/  # Example segmentations
    └── evaluation_metrics.txt
```

## 🔧 Configuration

### Training Hyperparameters

```python
# Model Configuration
IN_CHANNELS = 4              # FLAIR, T1, T1CE, T2
NUM_CLASSES = 4              # Background, Necrotic, Edema, Enhancing
FEAT_CHANNELS = [16, 32, 64, 128, 256]  # Feature channels per level
DROPOUT_RATE = 0.2           # Regularization strength

# Training Configuration
BATCH_SIZE = 2               # Limited by GPU memory
LEARNING_RATE = 1e-5         # Adam optimizer
WEIGHT_DECAY = 1e-5          # L2 regularization
NUM_EPOCHS = 16              # Maximum training epochs
PATIENCE = 3                 # Early stopping patience

# Data Configuration
PATCH_SIZE = (128, 128, 64)  # 3D patch dimensions
STRIDE = (64, 64, 32)        # Overlapping stride
BG_RATIO = 0.1               # Background to tumor patch ratio
```

### Inference Configuration

```python
# Patch-based Inference
PATCH_SIZE = (128, 128, 64)  # Must match training
STRIDE = (64, 64, 32)        # Overlap for smooth reconstruction
BATCH_SIZE = 4               # Inference batch size
```

## 🏥 Clinical Applications

### Potential Use Cases

- **Tumor Volume Quantification**: Automated measurement for treatment monitoring
- **Surgical Planning**: Precise tumor boundary delineation
- **Treatment Response Assessment**: Longitudinal volume change tracking
- **Clinical Trials**: Standardized endpoint evaluation
- **Screening Programs**: Large-scale automated analysis

### Clinical Validation Notes

⚠️ **Important**: This is a research implementation. Clinical use requires:
- Regulatory approval (FDA, CE marking, etc.)
- Extensive clinical validation
- Integration with PACS systems
- Radiologist oversight and validation

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update documentation as needed

## 📚 References

- **3D U-Net Paper**: Çiçek, Ö., et al. "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." MICCAI 2016.
- **BraTS Challenge**: Menze, B.H., et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)." IEEE TMI 2015.
- **TorchIO**: Pérez-García, F., et al. "TorchIO: a Python library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images." CMIG 2021.

## 🙏 Acknowledgments

- **BraTS Organizers** for providing the comprehensive dataset
- **Medical Imaging Community** for open-source tools and libraries
- **PyTorch Team** for the deep learning framework
- **Streamlit** for the web application framework

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult with qualified medical professionals for medical advice.

---

**Built with ❤️ for the medical imaging community**

For questions, issues, or suggestions, please open an issue or contact [your-email@domain.com](mailto:your-email@domain.com).
