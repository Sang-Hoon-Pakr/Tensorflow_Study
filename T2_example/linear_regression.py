import tensorflow as tf
import time
import numpy as np
import random
import matplotlib.pyplot as plt

input_dim = 2
output_dim = 1
learning_rate = 0.01

# 가중치 행렬입니다
w = tf.Variable(tf.random.uniform(shape=(input_dim, output_dim)))
# 편향 벡터입니다
b = tf.Variable(tf.zeros(shape=(output_dim,)))

def compute_predictions(features):
  return tf.matmul(features, w) + b

def compute_loss(labels, predictions):
  return tf.reduce_mean(tf.square(labels - predictions))

# 데코레이션을 통해서 속도 향상을 볼 수 있습니다. 
# 아래 @tf.function 에 주석을 주었다가 뺏다가 하며 시간을 측정해보세요  
# 데코레이션을 달지 않은경우 Epoch 당 0.074초
# 데코레이션을 단 경우 Epoch 당 0.061초가 소요되었습니다.

@tf.function
def train_on_batch(x, y):
  with tf.GradientTape() as tape:
    predictions = compute_predictions(x)
    loss = compute_loss(y, predictions)
    dloss_dw, dloss_db = tape.gradient(loss, [w, b])
  w.assign_sub(learning_rate * dloss_dw)
  b.assign_sub(learning_rate * dloss_db)
  return loss

# 데이터셋을 준비합니다
# 랜덤 데이터 
num_samples = 10000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3], cov=[[1, 0.5],[0.5, 1]], size=num_samples)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0], cov=[[1, 0.5],[0.5, 1]], size=num_samples)
features = np.vstack((negative_samples, positive_samples)).astype(np.float32)
labels = np.vstack((np.zeros((num_samples, 1), dtype='float32'),
                    np.ones((num_samples, 1), dtype='float32')))
plt.figure(1)
plt.scatter(features[:, 0], features[:, 1], c=labels[:, 0])
#plt.show()

print(np.shape(labels)) # 10000 , 1
print(np.shape(features)) # 10000, 2

# 데이터를 무작위로 섞습니다
random.Random(1337).shuffle(features)
random.Random(1337).shuffle(labels)

# 손쉽게 배치화된 반복을 위해, tf.data.Dataset 객체를 생성합니다
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(buffer_size=1024).batch(256)

t_start = time.time()
for epoch in range(10):
  for step, (x, y) in enumerate(dataset):
    # print("Shape of X", np.shape(x)) # 배치마다 256, 2가 들어감 여기서 2는 특징의 차원
    # print("Shape of Y", np.shape(y)) # 배치마다 256, 1이 들어감 
    loss = train_on_batch(x, y)
  print('Epoch %d: 마지막 배치의 손실값 = %.4f' % (epoch, float(loss)))
t_end = time.time() - t_start
print("Epoch당 소요 시간: %.3f 초" %(t_end/20))

predictions = compute_predictions(features)
plt.figure(2)
plt.scatter(features[:, 0], features[:, 1], c=predictions[:, 0] > 0.5)
plt.show()