# 🔍 AI Image Detector with Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![GUI](https://img.shields.io/badge/GUI-Tkinter-yellow)

A complete application for detecting AI-generated images using a hybrid model combining Deep Learning and traditional feature analysis.

## ✨ Features

### 🎯 Advanced Detection
- **Deep CNN model** with L2 regularization and Dropout
- **Hybrid analysis** combining deep learning + traditional features
- **Texture detection** with Local Binary Patterns (LBP)
- **Frequency analysis** using Fourier Transform (FFT)
- **Compression artifact** detection

### 🖥️ User Interface
- **Modern GUI** with Tkinter
- **Interactive visualization** of image features
- **Multiple tabs** (Analysis, Visualization, Settings)
- **Progress bars** for long operations
- **Complete logging** of analyses

### 🔧 Professional Tools
- **Custom training** with your own dataset
- **Batch analysis** with threading and progress tracking
- **Optional cross-validation** (5 folds)
- **Results export** to CSV/Excel
- **Cache management** to speed up analyses

### 📊 Metrics & Visualization
- **Detailed confusion matrices**
- **Comparative feature graphs**
- **Real-time overfitting tracking**
- **Complete classification reports**
- **Radar charts** of advanced features

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Automatic Installation
```bash
# Clone repository
git clone https://github.com/omarbadrani/ai-image-detector.git
cd ai-image-detector

# Install dependencies
pip install -r requirements.txt
```

### Manual Installation
```bash
pip install tensorflow pillow numpy opencv-python scikit-learn matplotlib seaborn pandas
```

## 📁 Project Structure

```
ai-image-detector/
├── detector.py               # Main application
├── requirements.txt          # Dependencies
├── config.json               # Configuration
├── ai_image_detector.h5      # Pre-trained model
├── best_model.h5            # Saved best model
├── logs/                    # Analysis logs
│   └── analysis_*.csv
├── dataset/                 # Recommended structure
│   ├── train/
│   │   ├── real/
│   │   └── ai/
│   └── test/
│       ├── real/
│       └── ai/
└── README.md                # This file
```

## 🎮 Usage

### Launch Application
```bash
python detector.py
```

### Quick Guide

1. **Single Image Analysis**:
   - Click "📁 Select Image"
   - Click "🔍 Analyze"
   - View detailed results

2. **Model Training**:
   - Click "🎓 Train Model"
   - Select real and AI image folders
   - Configure training parameters
   - Start training

3. **Batch Analysis**:
   - Click "📂 Analyze Folder"
   - Select folder containing images
   - Track real-time progress
   - Export results

### Supported Image Formats
- JPG/JPEG
- PNG
- BMP
- TIFF
- WebP

## 🧠 Technical Architecture

### Deep Learning Model
```python
Sequential([
    Augmentation Layer,
    Conv2D(32) + BatchNorm + Dropout(0.3),
    Conv2D(64) + BatchNorm + Dropout(0.3),
    Conv2D(128) + BatchNorm + Dropout(0.3),
    GlobalAveragePooling2D(),
    Dense(128) + Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### Analyzed Features
1. **Texture**: LBP, entropy, contrast
2. **Color**: Variance, LAB coherence
3. **Frequency**: FFT analysis
4. **Edges**: Density, quality
5. **Artifacts**: Compression, noise

## 📊 Performance

### Typical Metrics
| Metric | Value | Description |
|--------|-------|-------------|
| Training Accuracy | 98-99% | Performance on known data |
| Validation Accuracy | 75-85% | Performance on new data |
| Analysis Time | 1-3s/image | Hardware dependent |
| Model Size | ~15MB | Compressed .h5 file |

### Generalization Improvements
- **Early Stopping**: Automatic stop to prevent overfitting
- **LR Reduction**: Dynamic learning rate adjustment
- **Cross-Validation**: 5 folds for robustness
- **Data Augmentation**: Random transformations

## 🔧 Configuration

### config.json File
```json
{
    "img_size": [128, 128],
    "dropout_rate": 0.3,
    "l2_reg": 0.001,
    "batch_size": 32,
    "epochs": 30,
    "use_early_stopping": true,
    "early_stopping_patience": 10
}
```

## 📈 Results & Visualization

Application generates several visualization types:
1. **Bar charts**: Main features
2. **Radar plot**: Advanced features
3. **Confusion matrices**: Model performance
4. **Learning curves**: Overfitting tracking

## 🔧 Troubleshooting

### Common Issues:
1. **Memory error**: Reduce batch size
2. **Missing imports**: Run `pip install --upgrade -r requirements.txt`
3. **Model not loaded**: Delete and recreate `ai_image_detector.h5`

### Logs & Debug
- Logs saved in `logs/` folder
- Each analysis generates timestamped CSV file
- Errors captured and displayed in interface

## 📄 License
MIT License - see LICENSE file for details.

## 👤 Author
**omar badrani**  
- GitHub: https://github.com/omarbadrani  
- Email: omarbadrani770@gmail.com

---

⭐ **If this project is useful, please star the repository!** ⭐

---

**Version**: 1.0.0  
**Python**: 3.8+  
**OS**: Windows, Linux, macOS
