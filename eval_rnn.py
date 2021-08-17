from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import load_dataset #mizno's code
import numpy as np
from os.path import join
from sklearn.model_selection import StratifiedKFold

#this is also miznos's code
def convert_index_hotlabel(labels_index, n_classes):
	# converting from index to hot_lables
	hot_labels = [] # for def_convert_index_hotlabel()
	for lb in labels_index:
		hot_vector = [0] * n_classes
		# hot_vector = [0, 0, 0, 0, 0]
		i = int(lb) # convert from str(label) to int(label)
		hot_vector[i] = 1
		# hot_vector = [1, 0, 0, 0, 0]
		hot_labels.append(hot_vector)
	# print('mizno, ' + hot_labels)
	return np.array(hot_labels)


'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because WBC image shape is 128*128*3, we will then
handle 128*3 sequences of 128 steps for every sample.
'''
RESUME= False
SEED=123
# Training Parameters

# Network Parameters
num_input = 128*3 # MNIST data input (img shape: 28*28)
timesteps = 128 # timesteps
num_hidden = 32 # hidden layer num of features
num_classes = 5 # MNIST total classes (0-9 digits)

logdir = './log'
# tf Graph input



def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


class Model:
    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        self.X = tf.placeholder("float", [None, timesteps, num_input])
        self.Y = tf.placeholder("float", [None, num_classes])

        # Define weights
        self.weights = {
            'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([num_classes]))
        }       
        
        self.logits = RNN(self.X,self.weights,self.biases)
        self.prediction = tf.nn.softmax(self.logits)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=self.logits, labels=self.Y))

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.increment_global_step = tf.assign_add(self.global_step,1,
                                            name = 'increment_global_step')

            
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # for summary
        self.accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')


    def predict(self, x_test):
        predictions = np.empty((0,1))
        batch_size= 8 
        for i in range(0,x_test.shape[0],batch_size):
            x_batch = x_test[i:i+batch_size]
            prediction_i =  self.sess.run(self.prediction,feed_dict={self.X:x_batch})
            predictions = np.concatenate((predictions,np.argmax(prediction_i,axis=1)[:,np.newaxis]))
        print(predictions[:2,:])
        return predictions

    def get_accuracy(self,X,Y):
        #predictions = self.predict(X)
        #correct_pred = (predictions==Y)
        #acc = np.mean(correct_pred)
        acc = self.sess.run(self.accuracy,feed_dict={self.X:X,self.Y:Y})
        return acc


# Initialize the variables (i.e. assign their default value)
data_X,data_Y = load_dataset.main('/home/jumabek/data/wbc/new_5_crop_128_128')
data = np.concatenate((data_X,data_Y[:,np.newaxis]),axis=1)
np.random.seed(seed=SEED)

K=10
KFold_indices = []
stratifiedkfold = StratifiedKFold(n_splits=K,random_state=SEED,shuffle=True)
for train_index, test_index in stratifiedkfold.split(data_X, data_Y):
    KFold_indices.append((train_index,test_index))
    
with tf.Session() as sess:
    k=0
    #modeldir = 'models/lr_0.01_bs_64_h_32_s_50_ts_100000_k_0'
    modeldir = 'models/lr_0.01_bs_64_h_32_s_50_ts_100000'

    train_index, test_index = KFold_indices[k]
    
    X_train = data_X[train_index]
    Y_train = data_Y[train_index]
    print(X_train.shape, np.unique(Y_train))
    Y_train = convert_index_hotlabel(Y_train,num_classes)
    print(Y_train.shape,Y_train[:2,:]) 
    X_test = data_X[test_index]
    Y_test = data_Y[test_index]
    print(X_test.shape,np.unique(Y_test))
    Y_test = convert_index_hotlabel(Y_test,num_classes)
    print(Y_test.shape, Y_test[:2,:])
    
    train_size = Y_train.shape[0]
    test_size = Y_test.shape[0]
    X_train = X_train.reshape((train_size,timesteps,num_input))
    X_test = X_test.reshape((test_size,timesteps,num_input))

    model = Model(sess,'model')
    saver = tf.train.Saver() 
    saved_path = join(modeldir,"best.ckpt")
    saver.restore(sess,saved_path)

    loaded_step = sess.run(model.global_step)
    sess.run(tf.global_variables_initializer())
    print('best step: ', loaded_step)
    
    test_acc= model.get_accuracy(X_test,Y_test)
    print('Test Acc:',test_acc)
    
    train_acc = model.get_accuracy(X_train,Y_train)
    print("Train Accuracy:", train_acc)

