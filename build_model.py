from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, BatchNormalization, TimeDistributed, \
    Flatten, Input, GlobalAveragePooling2D, Reshape


def build_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Specify the input shape as (40, 40, 1)

    # Convolutional Layers
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    # Adding GlobalAveragePooling2D to reduce dimensions before LSTM
    model.add(GlobalAveragePooling2D())

    # Expanding dimensions to use in LSTM
    model.add(Reshape((1, 512)))

    # LSTM Layers
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))

    # Fully Connected Layer
    model.add(Dense(num_classes, activation='softmax'))

    return model