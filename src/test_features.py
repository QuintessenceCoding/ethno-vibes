# test_features.py
from extract_features import extract_face_features



features = extract_face_features("selfie.jpg")

if features is not None:
    print("Feature vector shape:", features.shape)
    print("First 10 features:", features[:10])
else:
    print("Face not found in image.")

img_path = "fairface-img-margin025-trainval/train/27584.jpg"  # <- use any actual file path from CSV
vec = extract_face_features(img_path)
print("✅ Feature vector shape:" if vec is not None else "❌ Face not detected.")