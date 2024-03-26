import cv2
# import pytesseract
# import json
import dlib
import numpy as np
import pytesseract
from PIL import Image


# Load the detector
detector = dlib.get_frontal_face_detector()

def extract_face(image_path, output_path, padding=35):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use the detector to find faces
    faces = detector(gray)
    for i, face in enumerate(faces):
        # Extract the face with some padding to make the image larger
        x1, y1, x2, y2 = face.left() - padding, face.top() - padding, face.right() + padding, face.bottom() + padding
        # Ensure the coordinates are within the image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1]-1, x2), min(img.shape[0]-1, y2)
        # Crop the face out of the image
        crop = img[y1:y2, x1:x2]
        # Save the image
        cv2.imwrite(f"{output_path}/face_{i}.jpg", crop)

def passportProcessing(image_path):
   # Reading the image
    img = cv2.imread(image_path)

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

    # Morphological operations to enhance text
    kernel = np.ones((1, 1), np.uint8)
    img_morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel)

    # Save the preprocessed image
    cv2.imwrite('preprocessed_image.jpg', img_morph)
    # img = Image.fromarray(thresh)
    if img is not None:
        text = pytesseract.image_to_string(img_morph,lang='eng+ara')
        texts = text.split('\n')
        # Applying the cleaning function to each text and filtering out single character or empty entries
        print(texts)
        filtered_texts_corrected = [clean_text(text) for text in texts if text.strip() and len(text.strip()) > 1]

        print(filtered_texts_corrected)
    else:
        print("Image not loaded correctly. Check the file path and file format: {image_path}")

def clean_text(text):
    return ''.join(char for char in text if char not in ('\u200f', '\u200e'))

def main():
    image_path = 'D:/image_processing/images/Passport-Iraq.jpg'  # replace with your image path
    file_path = 'output.json'  # replace with your output file path
    face_output_path = 'D:/image_processing/images'  # replace with your face output directory

    passportProcessing(image_path)
    extract_face(image_path,face_output_path,padding=35)

if __name__ == '__main__':
    main()
