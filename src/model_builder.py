from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models

def build_scratch_cnn(input_shape, num_classes):
    """
    CNN selon le diagramme, avec couche d'entrée déclarée :
      0. Input(shape)
      1. Conv2D(32) → MaxPooling2D
      2. Conv2D(64) → Conv2D(128)
      3. Flatten → Dense(512)
      4. Output (sigmoid ou softmax)
    """
    model = models.Sequential()

    # Couche d'entrée
    model.add(layers.Input(shape=input_shape))

    # Convolution initiale + pooling
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))  # Dropout pour éviter l'overfitting

    # Bloc convolutions
    model.add(layers.Conv2D(64,  (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Flatten + Dense
    # model.add(layers.Flatten())
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    # Couche de sortie
    if num_classes == 1:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'

    # Compilation
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=loss, metrics=['accuracy'])
    return model
