import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

gray_scale = 255
x_train /= gray_scale
x_test /= gray_scale


model = Sequential([
    Flatten(input_shape=(28, 28)),    # reshape 28 row * 28 column data to 28*28 rows
    Dense(256, activation='sigmoid'), # dense layer 1
    Dropout(0.2), # DropOut 20% on dense layer 1
    Dense(128, activation='sigmoid'), # dense layer 2
    Dropout(0.1), # DropOut 10% on dense layer 2
    Dense(10, activation='softmax'),  # dense layer 3
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

              
callbacks = [EarlyStopping(monitor='val_accuracy', patience=3)] # Early Stopping val_accuracy가 세번 이상 증가하지 않으면 early stopping 수행
model.fit(x_train, y_train, epochs=100, 
          batch_size=2000, validation_split = 0.2, callbacks=callbacks)

results = model.evaluate(x_test,  y_test, verbose = 0)

print('test loss, test acc:', results)