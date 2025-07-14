import os
import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model

from config        import MODEL_DIR, IMG_HEIGHT, IMG_WIDTH
from data_loader   import load_and_preprocess_image
from translate     import translate_label

@st.cache_resource
def load_resources():
    """Charge et met en cache le modèle + mapping index→label."""
    # On charge le meilleur checkpoint généré par ModelCheckpoint
    model_path   = MODEL_DIR / 'best_hybrid.keras'
    classes_path = MODEL_DIR / 'class_indices.pkl'

    # Vérifie l’existence et, le cas échéant, bascule sur le modèle final
    if not model_path.exists():
        model_path = MODEL_DIR / 'hybrid_final.keras'

    model = load_model(str(model_path))
    class_indices = joblib.load(str(classes_path))

    # inversion index→label_es
    idx2label_es = {v: k for k, v in class_indices.items()}
    # application de la traduction via translate_label()
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
    return idx2label[idx], float(preds[idx]), preds

def main():
    st.set_page_config(page_title="Image Classifier", layout="wide")
    st.title("🐾 Classificateur d'images d'animaux")
    st.write("Téléversez une image, je vous indique la classe prédite et la confiance.")

    model, idx2label = load_resources()

    uploaded_file = st.file_uploader(
        "Choisissez une image…",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Votre image", use_column_width=True)

        with st.spinner("Classification en cours…"):
            label, proba, preds = predict(image, model, idx2label)

        st.success(f"Classe prédite : **{label}** ({proba*100:.1f}% de confiance)")

        # Affichage du top-3
        st.write("Top 3 prédictions :")
        top3 = preds.argsort()[-3:][::-1]
        for i in top3:
            st.write(f"- {idx2label[i]} : {preds[i]*100:.1f}%")

    # Sidebar pour futur entraînement on-the-fly
    st.sidebar.header("Entraînement avancé")
    st.sidebar.write("→ bientôt : upload ZIP & entraînement intégré")

if __name__ == "__main__":
    main()
