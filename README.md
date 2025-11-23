Brain Tumor Classification Using CNN (PyTorch)

A deep learning project for classifying MRI images as “Tumor” or “No Tumor”.

🧠 Project Overview

This project builds a Convolutional Neural Network (CNN) using PyTorch to detect brain tumors from MRI images.
The dataset contains two classes:

Yes — MRI scans with tumor

No — MRI scans without tumor

This model uses:

Custom CNN architecture

Automatic flatten-size computation

Kaiming He initialization

TorchMetrics (Accuracy, Precision, Recall, F1)

Sigmoid final activation for binary classification

This project was developed in a Kaggle Notebook and exported to GitHub.

📁 Dataset

Dataset used:
Brain MRI Images for Brain Tumor Detection
Located on Kaggle at:
/kaggle/input/brain-mri-images-for-brain-tumor-detection/

Test images for testing inference helper was downloaded from this website, no copyright infringement intended:
https://www.researchgate.net/publication/341477427_Detection_of_Brain_Tumor_and_Identification_of_Tumor_Region_Using_Deep_Neural_Network_On_FMRI_Images

Dataset structure:

Brain Tumor Dataset/
    ├── yes/
    │     ├── Y1.jpg
    │     ├── Y2.jpg
    │     └── ...
    └── no/
          ├── N1.jpg
          ├── N2.jpg
          └── ...

🔧 Technologies Used

Python

PyTorch

TorchMetrics

torchvision

Kaggle Notebook

Matplotlib

NumPy

🏗️ Model Features

✔ Custom CNN with ELU activations
✔ Batch Normalization
✔ Dropout regularization
✔ Automatic flatten-size calculation
✔ Kaiming He initialization (Conv + Linear layers)
✔ Binary classification with sigmoid output
✔ Evaluation using TorchMetrics

📊 Training & Evaluation

Metrics used:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

🚀 How to Run Locally
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

2. Install dependencies
pip install torch torchvision torchmetrics matplotlib numpy

3. Download the dataset

Download from Kaggle and place inside:

project_root/
    └── data/
         ├── yes/
         └── no/

4. Run the notebook

Open Jupyter Notebook / VS Code and run:

Brain_Tumor_Classification.ipynb


or run Python script if you convert it.

📦 Repository Structure
.
├── README.md
├── brain_tumor_classification.ipynb
├── models/
│     └── tumor_cnn.py
├── data/
│     ├── yes/
│     └── no/
└── outputs/
      ├── training_plots/
      └── saved_model.pt

📚 Future Improvements

Add Grad-CAM heatmaps

Try transfer learning (ResNet50, EfficientNet)

Use data augmentation for better generalization

Convert notebook to a Python training script

🤝 Acknowledgements

Dataset by Kaggle contributors

PyTorch team

TorchMetrics library

🧑‍💻 Author

Den Bagayao
First deep learning project exploring CNN-based medical image classification.
