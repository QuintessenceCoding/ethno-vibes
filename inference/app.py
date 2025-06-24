import cv2
import numpy as np
import face_recognition
import joblib
import random
import streamlit as st



# === Paths ===
MODEL_PATH = "models/ethnicity_mlp_tuned.pkl"
PCA_PATH = "models/pca_transform.pkl"
IMAGE_PATH = "inference/selfie.jpg"

# === Load model and PCA ===
model = joblib.load(MODEL_PATH)
pca = joblib.load(PCA_PATH)

st.set_page_config(page_title="Ethnicity Predictor", page_icon="🌍")
st.title("🌟 AI Ethnicity Predictor")
st.markdown("Upload a selfie to discover your ethnicity mix — just for fun! 🧬📸")

uploaded_file = st.file_uploader("Upload a selfie", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    st.image(rgb, caption="Uploaded Selfie", use_container_width=True)

    if not face_encodings:
        st.error("😢 No face detected. Try another image with a clearer view of your face.")
    else:
        X = np.array([face_encodings[0]])
        X_reduced = pca.transform(X)
        probs = model.predict_proba(X_reduced)[0]
        classes = model.classes_

        percentages = {cls: round(prob * 100, 1) for cls, prob in zip(classes, probs)}
        sorted_ethnicities = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        st.subheader("📊 Ethnicity Breakdown:")
        for label, pct in sorted_ethnicities:
            st.write(f"- {label}: {pct}%")

        top_eth, top_pct = sorted_ethnicities[0]
        if top_pct > 70:
            st.success(f"🔍 Confident match: Definitely giving **{top_eth}** energy!")
        elif top_pct > 40:
            st.info(f"🤔 Strong leaning toward **{top_eth}**, but you’ve got a blend going on.")
        else:
            st.warning("🌍 Ethnically mysterious! A gorgeous mix from everywhere.")

        fun_lines = {
            "Indian": [
                "Namaste, bestie 🌸 Your desi glam is unmatched!",
                "Bollywood called, they want their star back 💃✨",
                "That glow? Totally made in India 🇮🇳"
            ],
            "White": [
                "Serving European summer realness 🍷🇫🇷",
                "Sunkissed and snow queen — you're the full aesthetic 🌞❄️",
                "That soft blonde energy is giving ✨ angelic ✨"
            ],
            "Black": [
                "Melanin magic poppin’ off! ✨🔥",
                "You're radiating royalty — Queen/King energy 👑🖤",
                "That skin tone? Chef’s kiss 💋🖤"
            ],
            "Latino_Hispanic": [
                "¡Ay caramba! That Latin heat is turning up 🔥💃",
                "Salsa in the soul and fire in the eyes 💃🌶️",
                "From Bogotá to Barcelona — you're worldwide 🔥"
            ],
            "Middle Eastern": [
                "Giving desert rose meets palace royalty 🌹🏜️",
                "Serving Middle Eastern mystery with a smoky eye ✨🧕",
                "Dates, gold, and glam — you’ve got it all 🕌💫"
            ],
            "East Asian": [
                "Your selfie is giving K-Drama lead vibes 🎬✨",
                "From Tokyo streets to Seoul beats — you're iconic 🌸🎧",
                "Too kawaii to handle 💖🫶"
            ],
            "Southeast Asian": [
                "Spicy, sweet, and glowing — total SEA goddess 🌶️🌺",
                "From Bali beauty to Manila muse — unstoppable ✨🌴",
                "Serving coconut milk glow and golden hour vibes 🥥☀️"
            ]
        }

        if top_eth in fun_lines:
            line = random.choice(fun_lines[top_eth])
        else:
            line = "You're a unique blend — no label can define you ✨🌎"

        st.subheader("💬 Fun Line:")
        st.info(line)
else:
    st.info("👆 Upload an image above to get started!")
