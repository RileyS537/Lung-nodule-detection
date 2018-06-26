from __future__ import division, print_function, absolute_import

import tflearn
import tensorflow as tf
import numpy as np

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
# n = 5 accuracy 75%
# n = 3 acc 80%
# n = 2  acc =82%
#n = 1  acc = 62%
n = 2

#solve OSError
import win_unicode_console
win_unicode_console.enable()


# Data loading
from tflearn.data_utils import image_preloader
X, Y = image_preloader(target_path = r'C:/Users/Administrator/Desktop/all_test/lungall', 
                        image_shape = (32, 32),mode ='folder',normalize=True,
                        grayscale=True, categorical_labels=True)

def my_func(x):
    x_list_to_array = np.array(x)
    x_s = x_list_to_array.reshape((-1, 32, 32, 1))

    #one channel to 3 channel
    a = x_s[:,:,:,0]
    a = a.reshape((-1, 32, 32, 1))
    x = np.concatenate((x_s, a), axis = 3)
    x = np.concatenate((x, a), axis = 3)
    return x

img_prep = tflearn.ImagePreprocessing()
img_prep.add_custom_preprocessing(my_func)

# Building Residual Network
net = tflearn.input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.resnext_block(net, n, 16, 32)
net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 32, 32)
net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
net = tflearn.resnext_block(net, n-1, 64, 32)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
loss = tflearn.fully_connected(net, 3, activation='softmax')
opt = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
network = tflearn.regression(loss, optimizer=opt, learning_rate=0.01,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(network, checkpoint_path='C:/Users/Administrator/Desktop/Resnet_test/model_resnext',
                    max_checkpoints=1, tensorboard_verbose=3,
                    tensorboard_dir='C:/Users/Administrator/Desktop/Resnet_test',
                    best_checkpoint_path='C:/Users/Administrator/Desktop/Resnet_test/model_resnet')

model.fit(X, Y, n_epoch=70, validation_set=0.1,
          snapshot_epoch=True, snapshot_step=200,
          show_metric=True, batch_size=64, shuffle=True)
model.save('C:/Users/Administrator/Desktop/Resnet_test/resnet_model')


#predict
X_test, Y_test = image_preloader(target_path = r'C:/Users/Administrator/Desktop/all_test/test', 
                        image_shape = (32, 32), mode ='folder',normalize=True,
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
print('test accuracy: %f'% acc_all)
