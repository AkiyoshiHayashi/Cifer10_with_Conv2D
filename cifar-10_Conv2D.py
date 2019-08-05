#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !conda install graphviz
# !conda install pydotplus 


# In[27]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import os
import pydot
import graphviz
from tensorflow.keras.utils import plot_model
cwd = os.getcwd()
print(cwd)
print(tf.__version__)


# In[28]:


session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config = session_config)
tf.keras.backend.set_session(sess)


# In[29]:


class_names = ['airplane', 'mobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse','ship', 'truck']


# In[30]:


def get_cifar10():
    cifar10 = keras.datasets.cifar10
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    return train_images, train_labels, test_images, test_labels


# In[31]:


def plot_cifar10(labels,images):
    global class_names
    plt.figure(figsize=(10,10))
    for i in range(0,25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[int(labels[i])])


# In[32]:


train_images, train_labels, test_images, test_labels = get_cifar10()


# In[33]:


plot_cifar10(train_labels,train_images)


# In[34]:


# For input neural network
train_images = train_images / 255.0
test_images = test_images / 255.0

img_rows, img_cols, img_channels = train_images.shape[1],train_images.shape[2],train_images.shape[3]


# In[35]:


print(img_rows,img_cols,img_channels)


# In[36]:


x_val = test_images[:3000]
partial_x_test = test_images[3000:]

y_val = test_labels[:3000]
partial_y_test = test_labels[3000:]


# In[47]:


Conv2D_model = keras.Sequential()
Conv2D_model.add(keras.layers.Conv2D(64, kernel_size=(3,3),
                              strides=(1,1),padding='same',
                              activation  = tf.nn.relu,
                              input_shape = (img_rows, img_cols, img_channels)))
Conv2D_model.add(keras.layers.Conv2D(128,(3,3),padding = 'same',activation = tf.nn.relu))
Conv2D_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

Conv2D_model.add(keras.layers.Conv2D(256, kernel_size=(3,3),
                              strides=(1,1),padding='same',
                              activation  = tf.nn.relu,
                              input_shape = (img_rows, img_cols, img_channels)))
Conv2D_model.add(keras.layers.Conv2D(512,(3,3),padding = 'same',activation = tf.nn.relu))
Conv2D_model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
'''
Conv2D_model.add(keras.layers.Conv2D(512, kernel_size=(3,3),
                              strides=(1,1),padding='same',
                              activation  = tf.nn.relu,
                              input_shape = (img_rows, img_cols, img_channels)))
Conv2D_model.add(keras.layers.Conv2D(1024,(3,3),padding = 'same',activation = tf.nn.relu))
'''
Conv2D_model.add(keras.layers.GlobalAveragePooling2D())
Conv2D_model.add(keras.layers.Dropout(0.25))
Conv2D_model.add(keras.layers.Dense(1024,activation='relu'))
# Conv2D_model.add(keras.layers.Dense(2028,activation='relu'))
Conv2D_model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

plot_model(Conv2D_model, to_file='model1.png', show_shapes=True)


# In[48]:


Conv2D_model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[49]:


Conv2D_model_history = Conv2D_model.fit(train_images, train_labels, 
                                        epochs=25,
                                        validation_data=(x_val, y_val),
                                        )


# In[50]:


test_loss, test_acc = Conv2D_model.evaluate(partial_x_test , partial_y_test )


# In[51]:


get_ipython().run_line_magic('matplotlib', 'inline')
history = Conv2D_model_history
history_dict = history.history

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[52]:


plt.clf()  
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# In[ ]:




