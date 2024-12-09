import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk, ImageEnhance
import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0
import cv2 as cv
import numpy as np

# Load the model for classification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0()
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, out_features=10, bias=True)  
)
model.load_state_dict(torch.load("animal_classifier.pth", map_location=device))  # Load pretrained weights
model.to(device)
model.eval() 

class_names = ["dog", "horse", "elephant", "butterfly", "chicken", "cat", "cow", "sheep", "spider", "squirrel"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Function to preprocess the image
def preprocess_image(img_path):
    """
    Preprocesses the input image by applying enhancements and augmentations.
    Steps:
        1. Load the image and convert it to RGB.
        2. Enhance contrast using histogram equalization in HSV space.
        3. Apply sharpening to improve details.
        4. Normalize and add noise for robustness.
    Args:
        img_path (str): Path to the input image file.
    Returns:
        PIL.Image: Original and augmented images as PIL.Image.
        torch.Tensor: Preprocessed and augmented image tensor.
    """
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Enhance contrast using histogram equalization in HSV
    img_hsv = cv.cvtColor(img_np, cv.COLOR_RGB2HSV)  # Convert to HSV color space
    img_hsv[:, :, 2] = cv.equalizeHist(img_hsv[:, :, 2])  # Equalize the value (brightness) channel
    img_enhanced = cv.cvtColor(img_hsv, cv.COLOR_HSV2RGB)  # Convert back to RGB

    # Apply sharpening to enhance image details
    img_pil = Image.fromarray(img_enhanced)  # Convert NumPy array back to PIL format
    enhancer = ImageEnhance.Sharpness(img_pil)
    img_sharpened = enhancer.enhance(2.0)  # Sharpen the image (factor of 2.0)

    img_tensor = transform(img_sharpened)

    return img, img_sharpened, img_tensor

# Function to classify the image
def classify_image(file_path):
    """
    Classifies an image by predicting the animal class using the loaded model.
    Args:
        file_path (str): Path to the image file.
    Returns:
        str: Predicted class name.
    """
    _, _, img_tensor = preprocess_image(file_path)  # Only get the tensor
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Add batch dimension and move to device
    with torch.inference_mode():  # Disable gradient computation for inference
        logits = model(img_tensor)  # Forward pass through the model
        probs = torch.softmax(logits, dim=1).squeeze()  # Compute probabilities
        predicted_idx = torch.argmax(probs).item()  # Get the index of the highest probability
    return class_names[predicted_idx]

def open_file():
    """
    Opens a file dialog to allow the user to upload an image and classify it.
    Updates the GUI with the uploaded image and predicted class.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")]) 
    if file_path:
        # Load the original and augmented images
        original_img, augmented_img, _ = preprocess_image(file_path)

        # Resize images for display
        original_img.thumbnail((400, 400))  
        augmented_img.thumbnail((400, 400))  

        # Convert images to PhotoImage format
        original_img_tk = ImageTk.PhotoImage(original_img)
        augmented_img_tk = ImageTk.PhotoImage(augmented_img)

        # Display the images horizontally in a grid
        original_img_label.config(image=original_img_tk)
        original_img_label.image = original_img_tk
        augmented_img_label.config(image=augmented_img_tk)
        augmented_img_label.image = augmented_img_tk

        # Show the labels below the images
        original_img_label_text.grid(row=1, column=0, pady=5)  
        augmented_img_label_text.grid(row=1, column=1, pady=5)  

        # Classify the image and display the result
        label = classify_image(file_path)
        result_label.config(text=f"Predicted: {label}")

root = tk.Tk()
root.title("Animal Classifier with Augmented Image Processing")

image_frame = tk.Frame(root)

original_img_label = Label(image_frame)
original_img_label.grid(row=0, column=0, padx=10)  

augmented_img_label = Label(image_frame)
augmented_img_label.grid(row=0, column=1, padx=10)  

original_img_label_text = Label(image_frame, text="Original Image", font=("Helvetica", 10))
augmented_img_label_text = Label(image_frame, text="Augmented Image", font=("Helvetica", 10))

result_label = Label(root, text="Upload an image to classify!", font=("Helvetica", 14))
result_label.pack(pady=20)  #

upload_btn = Button(root, text="Upload Image", command=open_file)
upload_btn.pack()

root.mainloop()
