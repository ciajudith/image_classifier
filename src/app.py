import os
import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model

from config        import MODEL_DIR, IMG_HEIGHT, IMG_WIDTH
from data_loader import load_and_preprocess_image
from translate    import translate_label

@st.cache_resource
def load_resources():
    """Charge et met en cache le modèle + mapping index→label."""
    model_path   = str(MODEL_DIR / 'animals10_best.h5')
    classes_path = str(MODEL_DIR / 'class_indices.pkl')

    model = load_model(model_path)
    class_indices = joblib.load(classes_path)
    # inversion index→label_es
    idx2label_es = {v: k for k, v in class_indices.items()}
    # application de la traduction
    idx2label = {
        idx: translate_label(label_es)
        for idx, label_es in idx2label_es.items()
    }
    return model, idx2label

def predict(image: Image.Image, model, idx2label):
    """Renvoie (label, proba, vecteur_probs) pour une PIL.Image."""
    x     = load_and_preprocess_image(image)
    preds = model.predict(x)[0]
    idx   = np.argmax(preds)
    return idx2label[idx], preds[idx], preds

def main():
    st.set_page_config(page_title="Image Classifier", layout="wide")
    st.title("🐾 Classificateur d'images d'animaux")
    st.write("Téléversez une image, je vous indique la classe prédite et la confiance.")

    model, idx2label = load_resources()

    uploaded_file = st.file_uploader(
        "Choisissez une image (PNG, JPG)…",
        type=["png","jpg","jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Votre image", use_column_width=True)

        with st.spinner("Classification en cours…"):
            label, proba, preds = predict(image, model, idx2label)

        st.success(f"Classe prédite : **{label}** ({proba*100:.1f}% de confiance)")

        # Top-3
        top_indices = preds.argsort()[-3:][::-1]
        st.write("Top 3 prédictions :")
        for i in top_indices:
            st.write(f"- {idx2label[i]} : {preds[i]*100:.1f}%")

    # Sidebar pour futur ré-entrainement
    st.sidebar.header("Entraînement avancé")
    st.sidebar.write("→ prochainement : upload ZIP & train on-the-fly")

if __name__ == "__main__":
    main()
