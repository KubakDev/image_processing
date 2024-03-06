import json
import cv2
import dlib
import numpy as np
from PIL import Image
import pytesseract

# Load the detector
detector = dlib.get_frontal_face_detector()

def preprocess_image(image_path):
    # Open the image file
    img = cv2.imread(image_path)
    # Convert the image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply threshold to get image with only black and white
    _, img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def extract_text_from_image(image):
    # Use pytesseract to convert the image into text
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def write_text_to_file(text, file_path):
    # Convert the extracted text to a dictionary
    data_dict = {}
    lines = text.split('\n')
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            data_dict[key] = value
    # Write the dictionary to a JSON file
    with open(file_path, 'w') as file:
        json.dump(data_dict, file)

def extract_face(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Use the detector to find faces
    faces = detector(gray)
    for i, face in enumerate(faces):
        # Extract the face
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        # Crop the face out of the image
        crop = np.copy(img[y1:y2, x1:x2])
        # Save the image
        cv2.imwrite(f"{output_path}/face_{i}.jpg", crop)

def main():
    image_path = 'C:/Users/hevar/OneDrive/Desktop/image_processing/images/passport.jpg'  # replace with your image path
    file_path = 'output.json'  # replace with your output file path
    face_output_path = 'C:/Users/hevar/OneDrive/Desktop/image_processing/images'  # replace with your face output directory

    image = preprocess_image(image_path)
    text = extract_text_from_image(image)
    write_text_to_file(text, file_path)
    extract_face(image_path, face_output_path)

if __name__ == '__main__':
    main()
