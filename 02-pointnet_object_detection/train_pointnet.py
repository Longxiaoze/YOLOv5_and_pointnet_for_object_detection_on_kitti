#!/usr/bin/env python
# coding: utf-8

# In[91]:


gpu_info = get_ipython().getoutput('nvidia-smi')
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)


# In[92]:


import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import pandas as pd


# In[93]:


print("Tensorflow Version: {}".format(tf.__version__))


# In[94]:


root = "./KITTI/object/object_training_datasets"
file_list = glob.glob(os.path.join(root,"*/*.txt"))
cls_names = [p.split("/")[7] for p in file_list]
cls_names = np.unique(cls_names)


# In[95]:


clsname_to_index = dict((name,index) for index,name in enumerate(cls_names))


# In[96]:


index_to_clsname = dict((index,name) for name,index in clsname_to_index.items())


# In[97]:


N = len(file_list)


# In[98]:


index_shuffle = np.random.permutation(N)


# In[99]:


file_list = np.asarray(file_list)[index_shuffle]


# In[100]:


train_list = file_list[:int(0.8*N)]
test_list = file_list[int(0.8*N):]


# In[101]:


train_paths = list(train_list)
test_paths = list(test_list)


# In[102]:


train_labels = [clsname_to_index.get(p.split("/")[7]) for p in train_paths]
test_labels = [clsname_to_index.get(p.split("/")[7]) for p in test_paths]


# In[103]:


train_datasets = tf.data.Dataset.from_tensor_slices((train_paths,train_labels))
test_datasets = tf.data.Dataset.from_tensor_slices((test_paths,test_labels))


# In[104]:


def load_train_fun(path,label):
  # 读取点云文件
  point_cloud_path = path.numpy()
  point_cloud_path = point_cloud_path.decode()
  point_cloud = pd.read_csv(point_cloud_path)
  point_cloud = np.array(point_cloud.iloc[:,0:6])

  # 归一化
  point_cloud[:,0:3] = point_cloud[:,0:3] - np.expand_dims(np.mean(point_cloud[:,0:3],0),0)
  dist = np.max(np.sqrt(np.sum(point_cloud[:,0:3]**2,axis=1)),0)
  if(dist > 0.0001):
    point_cloud[:,0:3] = point_cloud[:,0:3]/dist

  # 绕Z轴旋转随机角度
  theta = np.random.uniform(0,np.pi*2)
  rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
  point_cloud[:,[0,1]] = point_cloud[:,[0,1]].dot(rotation_matrix)
  point_cloud[:,[3,4]] = point_cloud[:,[3,4]].dot(rotation_matrix)

  point_cloud = tf.convert_to_tensor(point_cloud,dtype=tf.float64)

  # 添加噪声
  point_cloud += tf.random.uniform(point_cloud.shape, -0.005, 0.005, dtype=tf.float64)
  # 随机乱序
  point_cloud = tf.random.shuffle(point_cloud)

  label = tf.cast(label,dtype=tf.int64)
  return point_cloud,label


# In[105]:


def load_train(x,y):
    x, y = tf.py_function(load_train_fun, inp=[x, y], Tout=[tf.float64, tf.int64])
    return x,y


# In[106]:


def load_test_fun(path,label):
  # 读取点云文件
  point_cloud_path = path.numpy()
  point_cloud_path = point_cloud_path.decode()
  point_cloud = pd.read_csv(point_cloud_path)
  point_cloud = np.array(point_cloud.iloc[:,0:6])

  # 归一化
  point_cloud[:,0:3] = point_cloud[:,0:3] - np.expand_dims(np.mean(point_cloud[:,0:3],0),0)
  dist = np.max(np.sqrt(np.sum(point_cloud[:,0:3]**2,axis=1)),0)
  if(dist > 0.0001):
    point_cloud[:,0:3] = point_cloud[:,0:3]/dist

  point_cloud = tf.convert_to_tensor(point_cloud,dtype=tf.float64)
  label = tf.cast(label,dtype=tf.int64)
  return point_cloud,label


# In[107]:


def load_test(x,y):
    x, y = tf.py_function(load_test_fun, inp=[x, y], Tout=[tf.float64, tf.int64])
    return x,y


# In[108]:


train_datasets = train_datasets.shuffle(len(train_paths))
test_datasets = test_datasets.shuffle(len(test_paths))


# In[109]:


train_datasets = train_datasets.map(load_train)
test_datasets = test_datasets.map(load_test)


# In[110]:


NUM_POINTS = 64
NUM_CLASSES = 40
BATCH_SIZE = 16


# In[111]:


train_datasets = train_datasets.batch(BATCH_SIZE)
test_datasets = test_datasets.batch(BATCH_SIZE)


# In[112]:


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


# In[113]:


# 用来保证矩阵的秩为1
class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
      return {'num_features': int(self.num_features), 'l2reg': float(self.l2reg)}


# In[114]:


def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)
  
    x = conv_bn(inputs, 64)
    x = conv_bn(x, 128)
    x = conv_bn(x, 1024)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 512)
    x = dense_bn(x, 256)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


# In[115]:


inputs = keras.Input(shape=(NUM_POINTS, 6))

#x = tnet(inputs, 3)
x = conv_bn(inputs, 64)
x = conv_bn(x, 64)
#x = tnet(x, 64)
x = conv_bn(x, 64)
x = conv_bn(x, 128)
x = conv_bn(x, 1024)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 512)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()


# In[116]:


log_dir = os.path.join("logs",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


# In[117]:


tenserboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)


# In[118]:


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(train_datasets, epochs=14, validation_data=test_datasets,callbacks=[tenserboard_callback])


# In[119]:


model.save("pointnet_model.h5")


# In[120]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[121]:


index_to_clsname

