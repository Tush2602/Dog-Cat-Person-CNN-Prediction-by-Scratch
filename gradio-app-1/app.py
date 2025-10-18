import gradio as gr
import os
from datetime import datetime
from PIL import Image
from core.predict import ImageClassifier

from config import MODEL_DIR, OUTPUT_DIR, UPLOADED_IMG_DIR

model_path = MODEL_DIR
class_name = {0: 'Cat', 1: 'Dog', 2: 'person'}
classifier = ImageClassifier(model_path, class_name)

def classify_image(image):
    uploaded_folder = UPLOADED_IMG_DIR
    os.makedirs(uploaded_folder, exist_ok=True)
    image_path = f"{uploaded_folder}/uploaded_image_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.jpg"
    image.save(image_path)
    
    label, output_path = classifier.predict(image_path)

    return label, Image.open(output_path)

demo = gr.Interface(
    fn = classify_image,
    inputs =gr.Image(type="pil"),
    outputs = [gr.Textbox(label="Predictions"), gr.Image(label="Labeled Image")],
    title="Image Classification Gradio App",
    description = "Upload an image to classify it as Dog , Cat or Person"
)

if __name__=="__main__":
    demo.launch()


