import torch
import os
import sys
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from datetime import datetime

# print(os.getcwd())
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODEL_DIR, OUTPUT_DIR

class CustomCnnModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CustomCnnModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.conv_layers = nn.Sequential(
            #C1
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #C2
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #C3
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #C4
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self._to_linear= None
        self._get_conv_output(self.input_dim)

        self.fc_layers =nn.Sequential(
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

    def _get_conv_output(self, input_dim = 128):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_dim, input_dim)
            output = self.conv_layers(dummy_input)
            self._to_linear = output.view(1, -1).size(1)
            

    def forward(self, X):
        X = self.conv_layers(X)
        X = X.view(X.size(0), -1)
        X = self.fc_layers(X)
        return X 

class ImageClassifier:
    def __init__(self, model_path, class_name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CustomCnnModel(input_dim=128, num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location = self.device))
        self.model.eval()
        if class_name:
            self.class_name= {0: 'Cat', 1: 'Dog', 2: 'person'}
        else:
            self.class_name = class_name

        # transformation
        self.transform = transforms.Compose(
                                        [
                                        transforms.Resize((128, 128)),    # Reducing quality to speed up training
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                        
                                        ]
                                    )        

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            _, predicted = torch.max(output, 1)

            label = self.class_name[predicted.item()]


            img = cv2.imread(image_path)
            h, w, _ = img.shape
            font_scale = w / 500 
            thickness = max(2, w // 200) 
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = text_size[1] + 25

            cv2.putText(img, label, (text_x+2, text_y+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+1)
            cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
            output_path = OUTPUT_DIR
            os.makedirs(output_path, exist_ok=True)
            output_path = f"{output_path}/output_image_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
            cv2.imwrite(output_path, img)

            return label, output_path



    