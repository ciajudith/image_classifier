import json
import zipfile
from io import BytesIO

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import ConfusionMatrixDisplay

from config import MODEL_DIR
from data_loader import load_and_preprocess_image
from streamlit_live_metrics_callback import StreamlitLiveMetricsCallback
from train import train_with_zip
from f1score import F1Score  # Add this import


st.set_page_config(page_title="Classificateur d'Images", layout="wide")


def load_trained_model(model_name='hybrid_final.keras'):
    model_path = MODEL_DIR / model_name
    classes_path = MODEL_DIR / 'class_indices.pkl'

    if not model_path.exists() or not classes_path.exists():
        return None, None

    try:
        model = load_model(str(model_path), custom_objects={'F1Score': F1Score})
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None
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
            z = zipfile.ZipFile(BytesIO(uploaded_zip.getvalue()))
            # On collecte le premier composant de chaque chemin (les dossiers racine)
            classes = sorted({
                path.split('/')[0]
                for path in z.namelist()
                if '/' in path and not path.endswith('/')
            })
            st.markdown("**Classes détectées dans votre dataset :**")
            st.write("🔸 " + "\n🔸 ".join(classes))
            st.markdown("---")
            if st.button("Démarrer l’entraînement"):
                callback = StreamlitLiveMetricsCallback(total_epochs=epochs)
                zip_path = MODEL_DIR / "user_dataset.zip"
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())

                metrics, class_names = train_with_zip(
                    zip_path=zip_path,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    val_split=val_split,
                    extra_callbacks=[callback]
                )

                st.success("Entraînement terminé !")

    with tab2:
        st.header("Évaluation & Métriques")
        acc_img = MODEL_DIR / "accuracy.png"
        prec_img  = MODEL_DIR / "precision.png"
        rec_img = MODEL_DIR / "recall.png"
        loss_img = MODEL_DIR / "loss.png"
        report_path = MODEL_DIR / "classification_report.json"
        cm_path = MODEL_DIR / "confusion_matrix.npy"

        if acc_img.exists() and prec_img.exists() and rec_img.exists() and loss_img.exists():
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Exactitude (Accuracy)")
                st.image(str(acc_img), caption="Train & Validation Accuracy", use_container_width=True)

                st.subheader("Précision (Precision)")
                st.image(str(prec_img), caption="Train & Validation Precision", use_container_width=True)

            with col2:
                st.subheader("Rappel (Recall)")
                st.image(str(rec_img), caption="Train & Validation Recall", use_container_width=True)

                st.subheader("Perte (Loss)")
                st.image(str(loss_img), caption="Train & Validation Loss", use_container_width=True)
        else:
            st.info("Pas de courbes de training disponibles – lancez d’abord l’entraînement.")

        if report_path.exists() and cm_path.exists():
            st.markdown("---")
            col11, col22 = st.columns(2)
            report = json.loads(report_path.read_text())

            with col11:
                st.subheader("Matrice de confusion")
                labels = [cls for cls in report.keys() if cls not in ["accuracy", "macro avg", "weighted avg"]]
                cm = np.load(cm_path)
                fig, ax = plt.subplots(figsize=(7, 7))
                disp = ConfusionMatrixDisplay(cm, display_labels=labels)
                disp.plot(cmap="Blues", ax=ax, colorbar=False)
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_yticklabels(labels, rotation=0)
                st.pyplot(fig)

            with col22:
                st.subheader("Rapport de classification")

                for cls, m in report.items():
                    if cls in ["accuracy", "macro avg", "weighted avg"]:
                        continue
                    st.write(
                        f"**{cls}** : "
                        f"precision={m['precision']:.2f}, "
                        f"recall={m['recall']:.2f}, "
                        f"f1-score={m['f1-score']:.2f}, "
                        f"support={int(m['support'])}"
                    )
        else:
            st.info("Pas de rapport/matrice disponible – ré-entraînez le modèle pour les générer.")

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