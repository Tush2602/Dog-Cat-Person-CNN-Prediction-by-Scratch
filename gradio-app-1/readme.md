# 🧠 CNN Image Classification – Cat, Dog & Person

This project is a simple yet effective **Convolutional Neural Network (CNN)** built using **PyTorch** to classify images into **Cat**, **Dog**, and **Person** categories.  
It was created to learn image classification, model deployment, and real-time prediction with **Gradio**.

---

## 🚀 Project Overview

The goal of this project was to design, train, and deploy a CNN model that can automatically recognize images of cats, dogs, and humans.  
After training for 50 epochs, the model achieved a solid **83.50% test accuracy** 🎯.
Version --> Python 3.11.13

---

## 🧩 Features

- Custom CNN architecture built with **PyTorch**
- Runs on **CPU or GPU (CUDA)**
- Organized modular structure for easy updates
- Integrated **Gradio** interface for testing predictions
- Well-structured dataset and output folders

---

## 🛠️ Setup & Requirements

- Python 3.11.13
- PyTorch (specify version if needed)
- Gradio
- Other dependencies in `requirements.txt`

---

## 📂 Project Structure
DOG_CAT_PERSON_CNN_Classification_Model/
├── gradio-app-1/
│   ├── core/
│   │   ├── notebook/
│   │   │   ├── Classification_dataset/
│   │   │   ├── cnn_model_50_epoch.pth
│   │   │   └── main.ipynb
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── Output_Folder/
│   │   └── output_image_.jpg
│   ├── Uploaded_Image_Folder/
│   │   └── uploaded_image_.jpg
│   ├── app.py
│   ├── config.py
│   ├── README.md
│   ├── screenshot _of _working_app.png
│   └── requirements.txt

---

## ⚙️ How to Run the Project

1. **Install dependencies**
```
   pip install -r requirements.txt
```

2. **Run the Gradio app**
```
    python app.py
```

3. Upload an image (cat, dog, or person) in the Gradio interface to see predictions instantly.



![Gradio Interface Screenshot](<screenshot _of _working_app.png>)



| Metric             |     Value     |
| :----------------- | :-----------: |
| **Test Accuracy**  | 🟢 **83.50%** |
| **Optimizer**      |      Adam     |
| **Loss Function**  | Cross Entropy |
| **Epochs Trained** |       50      |


Hope you enjoy testing this model! Feel free to contribute or raise issues. ❤️



