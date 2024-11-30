import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
import os

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0()
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, out_features=10, bias=True)
)
model.load_state_dict(torch.load("animal_classifier.pth", map_location=device))
model.to(device)
model.eval()

# Class names
class_names = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def classify_image(file_path):
    img = Image.open(file_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(img_transformed)
        probs = torch.softmax(logits, dim=1).squeeze()
        predicted_idx = torch.argmax(probs).item()
    return class_names[predicted_idx]

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

        # Classify the image
        label = classify_image(file_path)
        result_label.config(text=f"Predicted: {label}")

# GUI setup
root = tk.Tk()
root.title("Animal Classifier")

img_label = Label(root)
img_label.pack()

result_label = Label(root, text="Upload an image to classify!", font=("Helvetica", 14))
result_label.pack()

upload_btn = Button(root, text="Upload Image", command=open_file)
upload_btn.pack()

root.mainloop()
