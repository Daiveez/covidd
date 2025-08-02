## 🩺 Chest X-ray CNN Classifier

A Convolutional Neural Network (CNN) built with PyTorch to classify chest X-ray images into three categories: **COVID-19**, **Viral Pneumonia**, and **Normal**. This project applies deep learning techniques to support medical diagnosis through image classification.

---

### 📂 Project Structure

```
chest_xray_cnn/
├── data/                     # Datasets (organized into train/val/test)
├── models/                   # Saved models (.pth files)
├── outputs/                  # Logs, plots, predictions
├── src/                      # Training, testing, and model scripts
│   ├── train.py
│   ├── model.py
│   ├── utils.py
│   └── evaluate.py
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── visualize_metrics.ipynb   # Training curves & visualizations
```

---

### 📊 Dataset

The dataset used is a collection of labeled **chest X-ray images**, categorized into:

* `COVID-19`
* `Viral Pneumonia`
* `Normal`

The data is split into **training**, **validation**, and **test** sets. You can organize your folders like this:

```
data/
├── train/
│   ├── COVID/
│   ├── NORMAL/
│   └── VIRAL_PNEUMONIA/
├── val/
└── test/
```

---

### 🧠 Model Architecture

The CNN model includes:

* Convolutional layers with ReLU and MaxPooling
* Fully connected layers
* Dropout for regularization
* Softmax output layer for multi-class classification

Optionally, you can replace this with a pre-trained model (e.g., ResNet18) using transfer learning.

---

### ⚙️ How to Train

```bash
python src/train.py
```

You can configure:

* Epochs
* Learning rate
* Batch size
* Optimizer (e.g., Adam)
* Early stopping criteria

Training and validation loss/accuracy are tracked per epoch and stored for plotting.

---

### ✅ Features Implemented

* 📦 Data preprocessing and augmentation (e.g., resizing, normalization, rotation)
* 🧠 CNN built from scratch using `torch.nn`
* 🏁 Training and testing loops with accuracy/loss calculation
* 🛑 Early stopping to prevent overfitting
* 💾 Model saving (`.pth`) and reloading
* 📈 Loss and accuracy plots with fixed y-axis ranges

---

### 📉 Example Metrics Plot

Training and validation loss/accuracy curves:

```python
# Accuracy y-axis: 0.10 to 1.0, Loss y-axis: 0 to 1
from visualize_metrics import plot_curves
plot_curves(...)
```

---

### 💾 Save/Load the Model

```python
# Save weights
torch.save(model.state_dict(), "models/covid_classifier.pth")

# Load later
model.load_state_dict(torch.load("models/covid_classifier.pth"))
```

---

### 📌 Requirements

* Python 3.8+
* PyTorch
* torchvision
* matplotlib
* numpy
* scikit-learn
* tqdm

Install with:

```bash
pip install -r requirements.txt
```

---

### 🧪 Future Improvements

* Replace CNN with transfer learning (ResNet / DenseNet)
* Add Grad-CAM or saliency maps for model explainability
* Web app deployment (Flask or Streamlit)
* Hyperparameter tuning

---

### 🙏 Acknowledgments

Dataset courtesy of public chest X-ray collections related to COVID-19 research. This project is intended for educational and research purposes only and **not for clinical use**.
