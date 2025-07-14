import os
import joblib
import matplotlib.pyplot as plt

from config        import DATA_DIR, MODEL_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, EPOCHS
from data_loader   import get_data_generators
from model_builder import build_scratch_cnn
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_and_save(model, train_gen, val_gen):
    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=str(MODEL_DIR / 'animals10_best.h5'),
            save_best_only=True,
            monitor='val_accuracy'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # sauvegarde finale
    model.save(str(MODEL_DIR / 'animals10_final.h5'))
    joblib.dump(
        train_gen.class_indices,
        str(MODEL_DIR / 'class_indices.pkl')
    )
    return history

def plot_history(history):
    # Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'],    label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.show()

    # Loss
    plt.figure()
    plt.plot(history.history['loss'],    label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.show()

def main():
    # 1. Data
    train_gen, val_gen = get_data_generators(
        data_dir=DATA_DIR,
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    # 2. Modèle
    model = build_scratch_cnn(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=train_gen.num_classes
    )

    # 3. Entraînement
    history = train_and_save(model, train_gen, val_gen)

    # 4. Plots
    plot_history(history)

    print(f"✅ Entraînement terminé. Modèles dans : {MODEL_DIR}")

if __name__ == "__main__":
    main()