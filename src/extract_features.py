# extract_features.py
import face_recognition
import numpy as np

def extract_face_features(image_path):
    try:
        img = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) == 0:
            print("⚠️ No face found in the image.")
            return None

        return np.array(encodings[0])  # Use first face encoding
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None
