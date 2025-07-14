import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model

from config import MODEL_DIR, IMG_HEIGHT, IMG_WIDTH
from data_loader import load_and_preprocess_image
from translate import translate_label

@st.cache_resource
def load_resources():
    """Charge et met en cache le modèle et le mapping index→label."""
    model_path = MODEL_DIR / 'best_hybrid.keras'
    classes_path = MODEL_DIR / 'class_indices.pkl'
    if not model_path.exists():
        model_path = MODEL_DIR / 'hybrid_final.keras'
    model = load_model(str(model_path))
    class_indices = joblib.load(str(classes_path))
    idx2label_es = {v: k for k, v in class_indices.items()}
    idx2label = {idx: translate_label(label_es) for idx, label_es in idx2label_es.items()}
    return model, idx2label

def predict(image: Image.Image, model, idx2label):
    """Renvoie (label, proba, vecteur_probs) pour une image PIL."""
    x = load_and_preprocess_image(image)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return idx2label[idx], float(preds[idx]), preds

def main():
    st.set_page_config(page_title="Classification d'image", layout="wide")

    # Centrage et largeur contrôlée
    left, center, right = st.columns([0.3, 3, 0.3])
    with center:
        st.markdown("## Classification d'image")
        st.markdown(
            "Dataset utilisé : [Animals-10 (Kaggle)](https://www.kaggle.com/datasets/alessiocorrado99/animals10)"
        )
        st.write("Téléversez une image pour obtenir la classe prédite et le score de confiance.")

        model, idx2label = load_resources()

        uploaded_file = st.file_uploader(
            "Choisissez une image…",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Votre image", use_container_width=True)

            with st.spinner("Classification en cours…"):
                label, proba, preds = predict(image, model, idx2label)

            st.success(f"Classe prédite : **{label}** ({proba*100:.1f}% de confiance)")

            st.markdown("**Top 3 prédictions :**")
            top3 = preds.argsort()[-3:][::-1]
            for i in top3:
                st.write(f"- {idx2label[i]} : {preds[i]*100:.1f}%")

    # Sidebar pour fonctionnalités à venir
    st.sidebar.header("Entraînement avancé")
    st.sidebar.info("🚧 Bientôt : téléversement d'un ZIP d'images et entraînement de votre propre modèle directement dans l'application.")

if __name__ == "__main__":
    main()