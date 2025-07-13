from tensorflow.keras import layers, models

def build_scratch_cnn(input_shape, num_classes):
    """
    CNN  selon le diagramme :
      1. Conv2D(32) → MaxPooling2D
      2. Conv2D(64) → Conv2D(128)
      3. Flatten → Dense(512)
      4. Output (sigmoid ou softmax)
    """
    model = models.Sequential()

    # 1. Convolution initiale + pooling
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))

    # 2. Bloc convolutions
    model.add(layers.Conv2D(64,  (3,3), activation='relu'))
    model.add(layers.Conv2D(128, (3,3), activation='relu'))

    # 3. Flatten + Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))

    # 4. Couche de sortie
    if num_classes == 1:
        model.add(layers.Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model
