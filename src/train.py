import os
import joblib
import matplotlib.pyplot as plt

from config             import MODEL_DIR, PHASE1_EPOCHS, PHASE2_EPOCHS, EPOCHS
from data_loader        import get_data_generators
from model_builder      import build_hybrid_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LambdaCallback
)
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import Adam

def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_gen, val_gen = get_data_generators()

    # 1. Instanciation du modèle
    num_classes = train_gen.num_classes
    model = build_hybrid_model(num_classes)


    # 2. Callbacks communs
    checkpoint = ModelCheckpoint(
        filepath=str(MODEL_DIR / 'best_hybrid.keras'),
        save_best_only=True,
        save_freq='epoch',
        monitor='val_loss'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    dump_pkl_cb = LambdaCallback(on_epoch_end=lambda epoch, logs:
    joblib.dump(
        train_gen.class_indices,
        str(MODEL_DIR / 'class_indices.pkl')
    ))

    callbacks = [checkpoint, reduce_lr, early_stop, dump_pkl_cb]

    # --- Phase 1 : entraîner SEULEMENT la tête ---
    for layer in model.layers:
        if hasattr(layer, 'trainable'):
            layer.trainable = layer.name.startswith('dense') or layer.name.startswith('batch_normalization')
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    joblib.dump(
        train_gen.class_indices,
        str(MODEL_DIR / 'class_indices.pkl')
    )

    # --- Phase 2 : fine-tuning des dernières couches de MobileNetV2 ---
    base = model.get_layer('mobilenetv2_1.00_224')  # ou 'mobilenetv2' selon votre version
    base.trainable = True
    # On gèle tout sauf les 20 derniers blocs
    for layer in base.layers[:-20]:
        layer.trainable = False

    # Scheduler CosineDecay sur PHASE2
    steps = EPOCHS * (train_gen.samples // train_gen.batch_size)
    schedule = CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=steps,
        alpha=1e-6
    )
    model.compile(
        optimizer=Adam(learning_rate=schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        initial_epoch=PHASE1_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Sauvegarde finale
    model.save(str(MODEL_DIR/'hybrid_final.keras'))
    joblib.dump(train_gen.class_indices, str(MODEL_DIR/'class_indices.pkl'))

    # Plot
    plt.plot(h1.history['val_accuracy'] + h2.history['val_accuracy'], label='val_acc')
    plt.plot(h1.history['val_loss']     + h2.history['val_loss'],     label='val_loss')
    plt.legend(); plt.show()

if __name__ == "__main__":
    train_and_save()
