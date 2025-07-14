import numpy as np
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)
from config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def get_data_generators(
    data_dir=DATA_DIR,
    img_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    val_split=0.2
):
    """Crée et renvoie les generators train / validation."""
    datagen = ImageDataGenerator(
        rescale=1. / 255,  # Normalise les pixels entre 0 et 1
        rotation_range=30,  # Applique une rotation aléatoire jusqu'à 30 degrés
        width_shift_range=0.1,  # Décale horizontalement l'image de 10% de la largeur
        height_shift_range=0.1,  # Décale verticalement l'image de 10% de la hauteur
        zoom_range=0.2,  # Applique un zoom aléatoire jusqu'à 20%
        horizontal_flip=True,  # Retourne l'image horizontalement de façon aléatoire
        fill_mode='nearest',  # Remplit les pixels vides avec la valeur la plus proche
        validation_split=val_split  # Fraction des données réservée à la validation
    )

    train_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=img_size,  # Redimensionne les images à la taille spécifiée
        batch_size=batch_size,  # Nombre d'images par lot
        class_mode='categorical',  # Les étiquettes sont encodées en one-hot
        subset='training',  # Utilise la partie entraînement du split
        shuffle=True  # Mélange les images à chaque époque
    )

    val_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=img_size,  # Redimensionne les images à la taille spécifiée
        batch_size=batch_size,  # Nombre d'images par lot
        class_mode='categorical',  # Les étiquettes sont encodées en one-hot
        subset='validation',  # Utilise la partie validation du split
        shuffle=False  # Ne mélange pas les images pour la validation
    )
    return train_gen, val_gen

def load_and_preprocess_image(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Charge une image depuis le disque, la redimensionne et normalise."""
    img = load_img(img_path, target_size=target_size)
    # Convertit l'image en tableau numpy et normalise les pixels entre 0 et 1
    x = img_to_array(img) / 255.0
    # Ajoute une dimension batch pour correspondre à l'entrée du modèle
    return np.expand_dims(x, axis=0)