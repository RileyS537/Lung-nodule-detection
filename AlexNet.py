from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
import numpy as np

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import image_preloader

# import tflearn.datasets.oxflower17 as oxflower17
# X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227)

import win_unicode_console
win_unicode_console.enable()

X, Y = image_preloader(target_path =r'C:/Users/Administrator/Desktop/all_test/lungall',
                        image_shape = (227,227), mode ='folder',normalize=True,grayscale=False,
                        categorical_labels=True)

def my_func(x):
    x_list_to_array = np.array(x)
    x_s = x_list_to_array.reshape((-1, 227, 227,1))

    #one channel to 3 channel
    a = x_s[:,:,:,0]
    a = a.reshape((-1,227,227,1))
    x = np.concatenate((x_s, a), axis = 3)
    x = np.concatenate((x, a), axis = 3)
    return x

img_prep = tflearn.ImagePreprocessing()
img_prep.add_custom_preprocessing(my_func)

network = input_data(shape=[None, 227, 227, 3], data_preprocessing=img_prep)

# Building 'AlexNet'
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)


loss = fully_connected(network, 3, activation='softmax')
network = regression(loss, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.01)
# Training
model = tflearn.DNN(network, checkpoint_path='C:/Users/Administrator/Desktop/alexnet',
                    max_checkpoints=1, tensorboard_verbose=3,
                    tensorboard_dir='C:/Users/Administrator/Desktop/alexnet',
                    best_checkpoint_path='C:/Users/Administrator/Desktop/alexnet/alex')

model.fit(X, Y, n_epoch=50, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=128, snapshot_epoch=True, snapshot_step=200)
model.save('C:/Users/Administrator/Desktop/alexnet/alex.model')

#predict
X_test, Y_test = image_preloader(target_path = r'C:/Users/Administrator/Desktop/all_test/test',
                        image_shape = (227,227), mode ='folder',normalize=True,
                        grayscale=True, categorical_labels=True)


X_test_array = np.array(X_test)
Y_test_array = np.array(Y_test)

num_test = len(Y_test_array)
groups = 20
image_per_group = int(num_test / 20)
acc_all = 0.0

for g in range(groups):

    X_test_mini = X_test_array[g*image_per_group:(g+1)*image_per_group,:,:]
    Y_test_mini = Y_test_array[g*image_per_group:(g+1)*image_per_group]

    probabilities = model.predict(X_test_mini)
    predict_label = model.predict_label(X_test_mini)

    print('group %d' % g)
    print(probabilities)
    print(predict_label)
   
    Y_predict = predict_label[:,0]
    Y_test_mini = Y_test_mini.argmax(axis = 1)
    acc_mini = (Y_predict == Y_test_mini).mean()
    print('group %d accuracy: %f' % (g, acc_mini))
    acc_all += acc_mini

acc_all = acc_all / groups
print('test accuracy: %f' % acc_all)
  

