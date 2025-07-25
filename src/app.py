import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from config import MODEL_DIR
from data_loader import load_and_preprocess_image
from streamlit_live_metrics_callback import StreamlitLiveMetricsCallback
from train import train_with_zip

st.set_page_config(page_title="🐾 Classificateur d'Images", layout="wide")


def load_trained_model(model_name='hybrid_final.keras'):
    model_path = MODEL_DIR / model_name
    classes_path = MODEL_DIR / 'class_indices.pkl'

    if not model_path.exists() or not classes_path.exists():
        return None, None

    model = load_model(str(model_path))
    class_indices = joblib.load(str(classes_path))
    idx2label = {v: k for k, v in class_indices.items()}
    return model, idx2label


def predict(image: Image.Image, model, idx2label):
    """Predict class, probability, and return output vector."""
    x = load_and_preprocess_image(image)
    preds = model.predict(x)[0]
    idx = np.argmax(preds)
    return idx2label[idx], float(preds[idx]), preds


def main():
    st.title(" Projet : Classification d'images")
    st.write(
        "Cette application permet :\n"
        "   1. d’entraîner un modèle sur un ZIP d’images structurées,\n"
        "   2. de visualiser en temps réel l’évolution des métriques,\n"
        "   3. de tester la classification sur une image au choix."
    )

    tab1, tab2, tab3 = st.tabs(["Entraînement", "Validation", "Test"])

    with tab1:
        st.header("Configuration de l’entraînement")
        uploaded_zip = st.file_uploader("Sélectionnez un ZIP de dataset", type=["zip"])

        col1, col2 = st.columns(2)
        with col1:
            epochs = st.number_input("Nombre d'époques", min_value=1, max_value=50, value=10, step=1)
            batch_size = st.select_slider("Taille de batch", options=[8, 16, 32, 64, 128], value=32)
        with col2:
            lr = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2,
                                 value=1e-3, format="%.6f")
            val_split = st.slider("Fraction validation", 0.1, 0.5, 0.2, step=0.05)

        if uploaded_zip:
            if st.button("Démarrer l’entraînement"):
                callback = StreamlitLiveMetricsCallback(total_epochs=epochs)
                zip_path = MODEL_DIR / "user_dataset.zip"
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())

                # Train and get history + class names
                metrics, class_names = train_with_zip(
                    zip_path=zip_path,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    val_split=val_split,
                    extra_callbacks=[callback]
                )

                st.success("Entraînement terminé !")
                st.write(f"Classes détectées : {', '.join(class_names)}")

                # Slider to select epoch to display
                history = metrics["history"]
                ep = st.slider("Afficher les métriques de l'époque", 1, epochs, 1)

                st.markdown(f"### Époque {ep} / {epochs}")
                st.write(f"- **Train accuracy** : {history['accuracy'][ep - 1]:.3f}")
                st.write(f"- **Train loss**      : {history['loss'][ep - 1]:.3f}")
                st.write(f"- **Val accuracy** : {history['val_accuracy'][ep - 1]:.3f}")
                st.write(f"- **Val loss**      : {history['val_loss'][ep - 1]:.3f}")

                # Graphs with marker on selected epoch
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(history['accuracy'], label='Train')
                axes[0].plot(history['val_accuracy'], label='Validation')
                axes[0].axvline(ep - 1, color='grey', linestyle='--')
                axes[0].set_title("Accuracy")
                axes[0].legend()
                # Loss
                axes[1].plot(history['loss'], label='Train')
                axes[1].plot(history['val_loss'], label='Validation')
                axes[1].axvline(ep - 1, color='grey', linestyle='--')
                axes[1].set_title("Loss")
                axes[1].legend()

                st.pyplot(fig)

    with tab2:
        st.header("Courbes de validation des métriques")
        acc_img = MODEL_DIR / "accuracy.png"
        loss_img = MODEL_DIR / "loss.png"
        if acc_img.exists() and loss_img.exists():
            st.image(str(acc_img), caption="Précision de validation", use_container_width=True)
            st.image(str(loss_img), caption="Loss de validation", use_container_width=True)
        else:
            st.info("Pas de métriques disponibles – entraîne d’abord un modèle.")

    with tab3:
        st.header("Tester une image")
        model, idx2label = load_trained_model()
        if model is None or idx2label is None:
            st.info("Aucun modèle entraîné disponible. Veuillez d'abord entraîner un modèle.")
        else:
            uploaded_img = st.file_uploader("Téléversez une image (png/jpg/jpeg)", type=["png", "jpg", "jpeg"])
            if uploaded_img:
                img = Image.open(uploaded_img).convert("RGB")
                st.image(img, caption="Votre image", use_container_width=True)
                with st.spinner("Classification…"):
                    label, proba, preds = predict(img, model, idx2label)
                st.success(f"Classe prédite : **{label}** ({proba * 100:.1f}% de confiance)")
                st.markdown("**Top 3 prédictions :**")
                for i in preds.argsort()[-3:][::-1]:
                    st.write(f"- {idx2label[i]} : {preds[i] * 100:.1f}%")


if __name__ == "__main__":
    main()