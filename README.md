# DIGIMAP 

# Animal Classifier with Augmented Image Processing

## MCO
- To access the executable file, kindly proceed to the [Google Drive link](https://drive.google.com/drive/folders/1Rqbfg1ip_Ftxvo6M5-1zIJe_NtRCOgyM?usp=drive_link) and download both files.
- Make sure to keep the `animal_classifier.pth` and `app.exe` in the same directory to ensure that the app will run.
- You can also run the application with `python app.py` on the terminal, just make sure that the `animal_classifer.pth` is on the same path as the `app.py`

## To build the app
- Install PyInstaller through `pip install pyinstaller`
- Create the executable file with `pyinstaller --onefile app.py`
- Make sure that `animal_classifier.pth` is placed on the same directory as the `app.exe` (in dist/)
- The `animal_classifier.pth` can be generated by running the whole `super resolution.ipynb` notebook

This application is a simple **image classification tool** built using **PyTorch** and **Tkinter**. It uses a pretrained **EfficientNet-B0** model to classify images of animals, and incorporates **data augmentation** techniques to enhance the images for better performance. The user can upload an image through the GUI, and the application will display the predicted animal class.

## Features

- **Image Classification**: Uses a pretrained EfficientNet-B0 model to predict the animal class in uploaded images.
- **Data Augmentation**: Applies various image enhancement techniques like contrast enhancement (via histogram equalization in HSV space) and sharpening to improve the image quality before classification.
- **GUI**: Built with Tkinter, providing an intuitive interface for users to upload images and view predictions.

## How It Works

- **Image Upload**: Users can upload an image via a file dialog.
- **Preprocessing and Augmentation**: The uploaded image is preprocessed with the following enhancements:
  - Contrast enhancement using histogram equalization in HSV color space.
  - Sharpening the image to enhance its details.
- **Classification**: The image is then passed through a pretrained EfficientNet-B0 model to classify the animal present in the image.
- **Result**: The predicted animal class is displayed below the image.

