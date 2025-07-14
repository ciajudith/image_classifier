import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from config import DATA_DIR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def get_data_generators(val_split=0.2, seed=42):
    """
    Utilise preprocess_input (–1,+1) pour MobileNetV2,
    + data-augmentation classique.
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=val_split
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=seed,
        shuffle=True
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=seed,
        shuffle=False
    )
    return train_gen, val_gen

def load_and_preprocess_image(img, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Pour l’app : prend un PIL.Image, resize, array et preprocess_input.
    """
    img = img.resize(target_size)
    x   = img_to_array(img)
    x   = preprocess_input(x)
    return np.expand_dims(x, axis=0)
