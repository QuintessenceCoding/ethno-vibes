import numpy as np
from collections import Counter

X = np.load("features.npy")
y = np.load("labels.npy")

print("✅ Feature shape:", X.shape)
print("🧠 Sample label distribution:")
print(Counter(y))
