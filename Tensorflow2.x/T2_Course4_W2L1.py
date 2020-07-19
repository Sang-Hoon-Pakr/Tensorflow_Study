import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

dataset = tf.data.Dataset.range(10)
for val in dataset:
   print(val.numpy())

print("==Stage 1 end==")
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1)
for window_dataset in dataset:
  for val in window_dataset:
    print(val.numpy(), end=" ")
  print()
print("==Stage 2 end==")

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
for window_dataset in dataset:
  for val in window_dataset:
    print(val.numpy(), end=" ")
  print()
print("==Stage 3 end==")

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
  print(window.numpy())
print("==Stage 4 end==")

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x,y in dataset:
  print(x.numpy(), y.numpy())
print("==Stage 5 end==")

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)

for x,y in dataset:
  print(x.numpy(), y.numpy())
print("==Stage 6 end==")

window_size=20
batch_size=32
shuffle_buffer_size=100
dataset = tf.data.Dataset.range(100)
dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
# dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.batch(batch_size).prefetch(1)
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
print("==Stage 7 end==")



