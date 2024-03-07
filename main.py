import cv2
# import pytesseract
# import json
import dlib
import numpy as np
import pytesseract
from PIL import Image


# Load the detector
detector = dlib.get_frontal_face_detector()
# def preprocess_image(image_path):
#     # Open the image file
#     img = cv2.imread(image_path)
#     # Check if the image was loaded correctly
#     if img is None:
#         raise ValueError(f"Image not loaded correctly. Check the file path and file format: {image_path}")
#     # Convert the image to gray scale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Apply threshold to get image with only black and white
#     _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return thresholded

# def extract_text_from_image(image):
#     # Use pytesseract to convert the image into text
#     custom_config = r'--oem 3 --psm 6'
#     d = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
#     return d

# def structure_data(data):
#     # Initialize an empty list to hold structured data
#     structured_data = {}
    
#     # Sort text blocks by their vertical position (y-coordinate)
#     sorted_blocks = sorted(data.items(), key=lambda item: item[1]['y'])
    
#     # Process each text block
#     for i, item in enumerate(sorted_blocks):
#         text_id, text_info = item
        
#         # Assuming labels are always in uppercase and values are not solely in uppercase
#         if text_info['text'].isupper() and len(text_info['text']) > 1:  # Likely a label
#             # Look ahead to the next text block for a potential value
#             if i+1 < len(sorted_blocks):
#                 next_text_id, next_text_info = sorted_blocks[i+1]
                
#                 # Check if the next block is horizontally aligned within a threshold
#                 if abs(text_info['x'] - next_text_info['x']) < 50:  # Threshold of 50 pixels
#                     label = text_info['text'].replace(":", "").strip()
#                     value = next_text_info['text'].strip()
#                     structured_data[label] = value
                    
#     return structured_data

# def write_structured_data_to_file(structured_data, file_path):
#     with open(file_path, 'w', encoding='utf-8') as file:
#         json.dump(structured_data, file, ensure_ascii=False, indent=4)

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
        filtered_texts_corrected = [clean_text(text) for text in texts if text.strip() and len(text.strip()) > 1]

        print(filtered_texts_corrected)
    else:
        print("Image not loaded correctly. Check the file path and file format: {image_path}")

# The user wants to only remove specific unicode markers and single characters, while keeping non-empty strings intact.
# Adjusting the code to filter based on the new criteria.

# Remove specific unicode markers from each text entry, filter out single character entries and empty strings
def clean_text(text):
    return ''.join(char for char in text if char not in ('\u200f', '\u200e'))

def main():
    image_path = 'D:/image_processing/images/passport.jpg'  # replace with your image path
    file_path = 'output.json'  # replace with your output file path
    face_output_path = 'D:/image_processing/images'  # replace with your face output directory

    passportProcessing(image_path)
    extract_face(image_path,face_output_path,padding=35)

    # image = preprocess_image(image_path)
    # text = extract_text_from_image(image)
    # write_structured_data_to_file(text, file_path)
    # extract_face(image_path, face_output_path)

if __name__ == '__main__':
    main()
