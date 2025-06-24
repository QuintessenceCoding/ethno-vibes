import os
import cv2
import numpy as np
import pandas as pd
import face_recognition
from tqdm import tqdm

# CONFIG
CSV_PATH = "fairface_label_train.csv"
IMG_DIR = "fairface-img-margin025-trainval/train"
MAX_IMAGES = 5000  # Adjust this if you want to increase/decrease dataset size

# Load and filter dataframe
df = pd.read_csv(CSV_PATH)
df = df[df['file'].notna()]
df = df[df['race'].notna()]
df = df.head(MAX_IMAGES)  # 💡 Limit image count for faster testing

features = []
labels = []

print(f"🚀 Starting feature extraction for {len(df)} images...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(IMG_DIR, row['file'].replace("train/", ""))
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

# Convert to arrays and save
features = np.array(features)
labels = np.array(labels)

# Filter classes with < 2 samples
unique, counts = np.unique(labels, return_counts=True)
valid_classes = unique[counts >= 2]
mask = np.isin(labels, valid_classes)
features = features[mask]
labels = labels[mask]

print(f"✅ Successfully extracted features from {len(features)} / {MAX_IMAGES} images")
print("🧠 Classes with ≥2 samples:")
for cls in np.unique(labels):
    count = np.sum(labels == cls)
    print(f" - {cls}: {count}")

np.save("features.npy", features)
np.save("labels.npy", labels)
print("✅ Saved features.npy and labels.npy")
