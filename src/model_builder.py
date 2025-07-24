from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2

from config import IMG_HEIGHT, IMG_WIDTH


def build_hybrid_model(num_classes,
                       scratch_filters=(32, 64, 128),
                       head_units=512,
                       l2_reg=1e-4):
    """
    Modèle hybride :
    - Branche scratch (quelques Conv→Pool)
    - Branche MobileNetV2 gelée
    - Fusion + Dense → sortie
    """
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    inputs = layers.Input(shape=input_shape)

    # --- Branche Scratch ---
    x1 = inputs
    for f in scratch_filters:
        x1 = layers.Conv2D(
            f, (3, 3), padding='same',
            activation='relu',
            kernel_regularizer=l2(l2_reg)
        )(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.MaxPooling2D()(x1)
        x1 = layers.Dropout(0.3)(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)

    # --- Branche pré-entraînée MobileNetV2 ---
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False
    x2 = base(inputs, training=False)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # --- Fusion ---
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(
        head_units,
        activation='relu',
        kernel_regularizer=l2(l2_reg)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # --- Sortie multiclasses ---
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model
