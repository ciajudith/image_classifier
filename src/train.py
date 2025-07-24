import os
import zipfile

import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam

from config import MODEL_DIR, DATA_DIR
from data_loader import get_data_generators
from model_builder import build_hybrid_model


def extract_dataset(zip_path, extract_to):
    if os.path.exists(extract_to):
        # Clean previous data
        for root, dirs, files in os.walk(extract_to, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def train_with_zip(zip_path, epochs, batch_size, lr, val_split, extra_callbacks=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    extract_dataset(zip_path, DATA_DIR)

    train_gen, val_gen = get_data_generators(
        val_split=val_split,
        seed=42,
        batch_size=batch_size
    )

    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())
    model = build_hybrid_model(num_classes)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    checkpoint = ModelCheckpoint(filepath=str(MODEL_DIR / 'best_hybrid.keras'), save_best_only=True, monitor='val_loss')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    callbacks = [checkpoint, reduce_lr, early_stop]
    if extra_callbacks:
        callbacks += extra_callbacks

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model.save(str(MODEL_DIR / 'hybrid_final.keras'))
    joblib.dump(train_gen.class_indices, str(MODEL_DIR / 'class_indices.pkl'))

    # Plot and save metrics
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Graph over Epochs')
    plt.legend()
    acc_path = MODEL_DIR / "accuracy.png"
    plt.savefig(acc_path)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Graph over Epochs')
    plt.legend()
    loss_path = MODEL_DIR / "loss.png"
    plt.savefig(loss_path)
    plt.close()

    # Precision
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Precision Graph over Epochs')
    plt.legend()
    prec_path = MODEL_DIR / "precision.png"
    plt.savefig(prec_path)
    plt.close()

    # Recall
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['recall'], label='Train Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Recall Graph over Epochs')
    plt.legend()
    rec_path = MODEL_DIR / "recall.png"
    plt.savefig(rec_path)
    plt.close()

    metrics = {
        "accuracy_plot": str(acc_path),
        "loss_plot": str(loss_path),
        "history": history.history
    }
    return metrics, class_names
