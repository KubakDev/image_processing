from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
from passporteye import read_mrz
from datetime import datetime
from appwrite_module import get_storage, get_database
from appwrite.input_file import InputFile
from appwrite.id import ID
import logging

app = Flask(__name__)
CORS(app)
storage = get_storage()
app.logger.setLevel(logging.INFO)

# Use Flask's instance path to construct the path to the uploads directory
UPLOAD_FOLDER = os.path.join("", "static/uploads")
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Assuming you have the haarcascade file in the same directory as your app
HAARCASCADE_PATH = "haarcascade_frontalface_default.xml"


def extract_face(filepath):
    try:
        img = cv2.imread(filepath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return None  # No face detected

        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        face_filepath = filepath.rsplit(".", 1)[0] + "_face.jpg"
        cv2.imwrite(face_filepath, face)
        return face_filepath.replace("\\", "/")
    except Exception as e:
        app.logger.info(f"Error in extract_face: {e}")
        return None


@app.route("/api/data", methods=["GET"])
def get_data():
    data = {"message": "Hello from Python backend!"}
    return jsonify(data)


@app.route("/api/process_passport", methods=["POST"])
def process_passport():
  try:
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Process the MRZ
        mrz_data = read_mrz(filepath)
        if mrz_data is None:
            return jsonify({"error": "Could not extract MRZ"}), 400
        mrz_dict = mrz_data.to_dict()

        # Apply fixes
        surname = mrz_dict.get("surname").lstrip("Q")  # Remove leading 'Q' if present
        names = mrz_dict.get("names").replace(
            "X", ""
        )  # Replace 'X' with space in names
        personal_number = mrz_dict.get("personal_number").replace(
            "<", ""
        )  # Remove '<' from personal number

        # Convert dates from YYMMDD to a more readable format
        dob = datetime.strptime(mrz_dict.get("date_of_birth"), "%y%m%d").strftime(
            "%Y-%m-%d"
        )
        exp_date = datetime.strptime(
            mrz_dict.get("expiration_date"), "%y%m%d"
        ).strftime("%Y-%m-%d")

        # Apply fixes and enhancements
        country_correction = {"ITR": "IRQ"}  # Example correction
        corrected_country = country_correction.get(
            mrz_dict.get("country"), mrz_dict.get("country")
        )

        # Exclude personal number if it's empty or contains only filler characters
        personal_number = mrz_dict.get("personal_number").replace("<", "")
        if all(c == "<" for c in mrz_dict.get("personal_number")):
            personal_number = ""

        # Convert dates from YYMMDD to a more readable format
        dob = datetime.strptime(mrz_dict.get("date_of_birth"), "%y%m%d").strftime(
            "%Y-%m-%d"
        )
        exp_date = datetime.strptime(mrz_dict.get("expiration_date"), "%y%m%d").strftime(
            "%Y-%m-%d"
        )
        # Process the face image
        face_filename = extract_face(filepath)
        result = storage.create_file('6602a873de0e9ff815f0', ID.unique(), InputFile.from_path(face_filename))
        file_url = f"https://cloud.appwrite.io/v1/storage/buckets/6602a873de0e9ff815f0/files/{result['$id']}/view?project=6602a79975c04c55b0a3"
        if face_filename:
            face_url = f"static/uploads/{face_filename}"  # Adjust path as necessary

        # Extracting passport info with corrections
        extracted_data = {
            "passport_number": mrz_dict.get("number"),
            "country": corrected_country,
            "surname": surname,
            "names": names,
            "nationality": mrz_dict.get("nationality"),
            "date_of_birth": dob,
            "sex": mrz_dict.get("sex"),
            "expiration_date": exp_date,
            "personal_number": personal_number,
        }

        # Include the face image URL if the face was processed
        if face_filename:
            extracted_data["face_image_url"] = file_url

        return jsonify(extracted_data)
  except Exception as e:
      return None


if __name__ == "__main__":
    app.run(port=5000, debug=True)
