#Conv2D and Maxpooling example
import tensorflow as tf

#Callback Example 
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train, x_test = x_train / 255.0, x_test / 255.0

callbacks = myCallback()

#내가 모르는 것일 수 있으나...
#Batch_size가 조절이 안되는 느낌... 
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPool2D(2, 2),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
  tf.keras.layers.MaxPool2D(2, 2),
  tf.keras.layers.Dropout(0.3),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
model.evaluate(x_test, y_test, verbose=1)
