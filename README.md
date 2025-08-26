# Brain Tumor Detection using Deep Learning

A comprehensive deep learning project for automatic brain tumor classification using Convolutional Neural Networks (CNN). This project can classify brain MRI images into four categories: Glioma, Meningioma, Pituitary, and No Tumor.

## 🧠 Project Overview

This project implements a CNN-based model for brain tumor detection and classification from MRI scans. The model achieves high accuracy in distinguishing between different types of brain tumors and normal brain tissue.

## ✨ Features

- **Multi-class Classification**: Classifies brain MRI images into 4 categories
- **High Accuracy**: Achieves 94.13% test accuracy
- **Data Augmentation**: Implements various image augmentation techniques
- **Comprehensive Evaluation**: Includes confusion matrix, ROC curves, and detailed metrics
- **Pre-trained Model**: Ready-to-use CNN model for inference

## 🏗️ Architecture

The CNN model consists of:
- **3 Convolutional Layers** with MaxPooling and BatchNormalization
- **Flatten Layer** for feature extraction
- **2 Dense Layers** with Dropout for regularization
- **Softmax Output** for multi-class classification

## 📊 Dataset

The model is trained on a dataset containing:
- **Training**: 5,712 images across 4 classes
- **Testing**: 1,311 images across 4 classes
- **Classes**: Glioma, Meningioma, Pituitary, No Tumor

## 🚀 Performance Metrics

- **Overall Accuracy**: 94.13%
- **Class-wise Performance**:
  - Glioma: 95% precision, 89% recall
  - Meningioma: 88% precision, 88% recall
  - No Tumor: 97% precision, 100% recall
  - Pituitary: 96% precision, 99% recall

## 📁 Project Structure

```
Brain-Tumor-Detection/
├── braintumour.ipynb          # Main Jupyter notebook
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore file
└── models/                    # Saved model files (if any)
```

## 🛠️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/7-dante-7/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**:
   ```bash
   jupyter notebook braintumour.ipynb
   ```

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.x
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 🔧 Usage

1. **Data Preparation**: Ensure your brain MRI dataset is organized in the following structure:
   ```
   Training/
   ├── glioma/
   ├── meningioma/
   ├── notumor/
   └── pituitary/
   
   Testing/
   ├── glioma/
   ├── meningioma/
   ├── notumor/
   └── pituitary/
   ```

2. **Model Training**: Run the notebook cells to train the CNN model
3. **Evaluation**: Use the provided evaluation functions to assess model performance
4. **Inference**: Load the trained model for new predictions

## 📈 Model Training

The model is trained with:
- **Epochs**: 50
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## 📊 Results Visualization

The project includes comprehensive visualization tools:
- Training vs. Validation accuracy/loss curves
- Confusion matrix heatmap
- Class-wise accuracy comparison
- ROC curves for each class

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**7-dante-7**

## 🙏 Acknowledgments

- Dataset providers for brain MRI images
- Open source community for deep learning frameworks
- Research community for CNN architectures and techniques

---

**Note**: This project is for educational and research purposes. Always consult medical professionals for actual medical diagnoses.
