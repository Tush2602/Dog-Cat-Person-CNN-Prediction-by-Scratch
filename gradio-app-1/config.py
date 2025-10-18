import os
import sys

CWD_DIR = os.getcwd()
BASE_DIR = os.path.join(CWD_DIR, "gradio-app-1")

MODEL_DIR = os.path.join(BASE_DIR, "core/notebook/cnn_model_50_epoch.pth")
# MODEL_DIR = r"C:\Users\Tushar\OneDrive\Desktop\Dog_Cat_Person_CNN_Classification Model\gradio-app-1\core\notebook\cnn_model_50_epoch.pth"
OUTPUT_DIR = os.path.join(BASE_DIR, "Output Folder")
UPLOADED_IMG_DIR = os.path.join(BASE_DIR, "Uploaded Image Folder")


#Notebook directories
NOTEBOOK_DIR = os.path.join(BASE_DIR, "core/notebook")
DATA_DIR = os.path.join(NOTEBOOK_DIR, "Classification_dataset_v3\images")
TRAINING_DIR = os.path.join(DATA_DIR, "train")
TESTING_DIR = os.path.join(DATA_DIR, "test")
CORE_OP_FOLDER = os.path.join(NOTEBOOK_DIR, "OP FOLDER")