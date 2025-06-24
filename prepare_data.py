import os
import cv2
import numpy as np
import pandas as pd
import face_recognition
from tqdm import tqdm

# CONFIG
CSV_PATH = "data/fairface_label_train.csv"
IMG_DIR = "data/fairface-img-margin025-trainval/train"
MAX_IMAGES = 5000  # Change this to process more or fewer images

# Load and filter dataframe
df = pd.read_csv(CSV_PATH)
df = df[df['file'].notna()]
df = df[df['race'].notna()]
df = df.head(MAX_IMAGES)

features = []
labels = []

print(f"🚀 Starting feature extraction for {len(df)} images...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    # Fix path: strip 'train/' if it's included in CSV
    filename = row['file'].replace("train/", "")
    img_path = os.path.join(IMG_DIR, filename)
    
    image = cv2.imread(img_path)
    if image is None:
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)

    if len(face_locations) == 0:
        continue

    encodings = face_recognition.face_encodings(rgb, face_locations)
    if len(encodings) == 0:
        continue

    features.append(encodings[0])
    labels.append(row['race'])

# Convert to arrays
features = np.array(features)
labels = np.array(labels)

# Filter out classes with <2 samples
unique, counts = np.unique(labels, return_counts=True)
valid_classes = unique[counts >= 2]
mask = np.isin(labels, valid_classes)
features = features[mask]
labels = labels[mask]

print(f"✅ Successfully extracted features from {len(features)} / {MAX_IMAGES} images")
print("🧠 Classes with ≥2 samples:")
for cls in np.unique(labels):
    print(f" - {cls}: {np.sum(labels == cls)}")

# Save to features/ directory
os.makedirs("features", exist_ok=True)
np.save("features/features.npy", features)
np.save("features/labels.npy", labels)
print("✅ Saved to features/features.npy and features/labels.npy")
