import os
import zipfile

import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay

from config import MODEL_DIR, DATA_DIR
from data_loader import get_data_generators
from model_builder import build_hybrid_model


def extract_dataset(zip_path, extract_to):
    if os.path.exists(extract_to):
        # suppression du contenu précédent
        for root, dirs, files in os.walk(extract_to, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)


def train_with_zip(
        zip_path,
        epochs,
        batch_size,
        lr,
        val_split,
        extra_callbacks=None
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    extract_dataset(zip_path, DATA_DIR)

    # Data
    train_gen, val_gen = get_data_generators(
        val_split=val_split,
        seed=42,
        batch_size=batch_size
    )

    num_classes = train_gen.num_classes
    class_names = list(train_gen.class_indices.keys())

    # Modèle hybride
    model = build_hybrid_model(num_classes)

    # Callbacks communs
    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_DIR / 'best_hybrid.keras'),
        save_best_only=True, monitor='val_loss'
    )
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )
    callbacks = [checkpoint, reduce_lr, early_stop]
    if extra_callbacks:
        callbacks += extra_callbacks

    # ─── Phase 1 : n’entraîne que la tête (Dense+BN) ────
    # freeze tout sauf Dense et BatchNorm
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization) \
                or layer.name.startswith('dense'):
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    phase1_epochs = epochs // 2
    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=phase1_epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Phase 2 : fine‑tuning du backbone
    # défreeze la base MobileNetV2
    base = next(l for l in model.layers if 'mobilenetv2' in l.name)
    base.trainable = True
    # gèle tout sauf les 20 derniers blocs
    for layer in base.layers[:-20]:
        layer.trainable = False

    # LR scheduling pour phase 2
    steps_per_epoch = train_gen.samples // train_gen.batch_size
    phase2_epochs = epochs - phase1_epochs
    cos_decay = CosineDecay(
        initial_learning_rate=lr * 0.1,
        decay_steps=phase2_epochs * steps_per_epoch
    )

    model.compile(
        optimizer=Adam(learning_rate=cos_decay),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        initial_epoch=phase1_epochs,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Sauvegarde du modèle final
    model.save(str(MODEL_DIR / 'hybrid_final.keras'))
    joblib.dump(train_gen.class_indices, str(MODEL_DIR / 'class_indices.pkl'))

    # Combinaison des historiques
    history = {}
    for k in h1.history:
        history[k] = h1.history[k] + h2.history[k]

    # Tracés & sauvegardes des métriques
    def save_plot(key, title, ylabel):
        plt.figure(figsize=(8, 4))
        plt.plot(history[key], label=f'Train {key}')
        plt.plot(history[f'val_{key}'], label=f'Val {key}')
        plt.title(f'{title} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        path = MODEL_DIR / f"{key}.png"
        plt.savefig(path)
        plt.close()
        return str(path)

    acc_path = save_plot('accuracy', 'Accuracy', 'Accuracy')
    prec_path = save_plot('precision', 'Precision', 'Precision')
    rec_path = save_plot('recall', 'Recall', 'Recall')
    loss_path = save_plot('loss', 'Loss', 'Loss')

    metrics = {
        "accuracy_plot": acc_path,
        "precision_plot": prec_path,
        "recall_plot": rec_path,
        "loss_plot": loss_path,
        "history": history
    }
    return metrics, class_names
