import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import os

# === Load features and labels
X = np.load("features/features.npy")
y = np.load("features/labels.npy")

# === Reduce dimensions with PCA
pca = PCA(n_components=50, random_state=42)
X_reduced = pca.fit_transform(X)

# Save PCA
os.makedirs("models", exist_ok=True)
joblib.dump(pca, "models/pca_transform.pkl")
print("✅ PCA saved at: models/pca_transform.pkl")

# === Define model and grid search
mlp = MLPClassifier(max_iter=500, random_state=42)

param_grid = {
    "hidden_layer_sizes": [(128,), (128, 64), (256, 128)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001],
    "learning_rate": ["adaptive", "constant"]
}

grid = GridSearchCV(mlp, param_grid, cv=3, verbose=2, n_jobs=-1)
grid.fit(X_reduced, y)

# === Save best model
joblib.dump(grid.best_estimator_, "models/ethnicity_mlp_tuned.pkl")
print("✅ MLP model saved at: models/ethnicity_mlp_tuned.pkl")

# === Report
y_pred = grid.predict(X_reduced)
print("\n📊 Classification Report:\n")
print(classification_report(y, y_pred, digits=2))
