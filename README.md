# 🌍 Ethnicity Fun App

A lighthearted AI app that predicts what ethnicity you might resemble based on your selfie! Just for fun — no serious claims or analysis here. 🧬📸

---

## ✨ Features

- 📸 Upload a selfie
- 🧠 Facial features extracted using `face_recognition`
- 🎯 Predictions using a tuned `MLPClassifier` (scikit-learn)
- 📊 Shows ethnicity breakdown in %
- 💬 Fun, Gen-Z style one-liner compliments based on the prediction

---

## 🚀 Demo

> Run locally:
````markdown
streamlit run app.py
````

Upload a selfie and see the magic happen! ✨

---

## 🧠 How it Works

* Trained on the [FairFace Dataset](https://github.com/joojs/fairface)
* Uses `face_recognition` to extract facial encodings
* Dimensionality reduction via PCA
* Predicts ethnicity using MLPClassifier

---

## ⚠️ Disclaimer

This app is **not meant to classify real ethnicity**.
It only shows which ethnic group your facial features most resemble **according to the model**, and is intended for entertainment and exploration.

---

## 📂 Project Structure

```
ethnicity-fun-app/
│
├── app.py                  # Streamlit web app
├── inference/              # Inference scripts and selfie image
├── models/                 # Trained model and PCA transformer
├── src/                    # Training & preprocessing scripts
├── data/                   # CSV label files
├── features/               # Saved face encodings (optional)
├── train_model.py          # Model training entry point
├── prepare_data.py         # Data preparation script
└── requirements.txt        # Python dependencies
```

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

You’ll need:

* Python 3.8+
* Streamlit
* scikit-learn
* face\_recognition
* OpenCV
* NumPy
* joblib

---

## 💡 Inspiration

Inspired by apps like Gradient and Face++ — but built for open-source fun, transparency, and no creepy tracking. 🙃

---

## 🧑‍💻 Author

**Ishika** – Made with love, curiosity, and chaos.

---

## ⭐️ Star this repo

If you enjoyed this project, please consider giving it a ⭐ on GitHub — it really helps!

```

