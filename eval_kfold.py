"""
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

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
row as a sequence of pixels. Because WBC image shape is 128*128*3px, we will then
handle 128*3 sequences of 128 steps for every sample.
'''
RESUME= False
SEED=123
# Training Parameters
learning_rate = 0.3 
training_steps = 100000
batch_size = 32 
validate_step = 50

# Network Parameters
num_input = 128*3 # MNIST data input (img shape: 28*28)
timesteps = 128 # timesteps
num_hidden = 16 # hidden layer num of features
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

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.increment_global_step = tf.assign_add(self.global_step,1,
                                            name = 'increment_global_step')

        self.train_op = self.optimizer.minimize(self.loss_op, global_step=self.global_step)
            
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # for summary
        self.accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
        self.loss_ph = tf.placeholder(tf.float32,shape=None, name='loss_summary')

        self.acc_summary = tf.summary.scalar('train_accuracy',self.accuracy_ph)
        self.loss_summary = tf.summary.scalar('loss', self.loss_ph)
        #self.performance_summaries = tf.summary.merge([acc_summary,loss_summary])

        self.test_acc_summary = tf.summary.scalar('test_accuracy',self.accuracy_ph)


    def predict_for_test(self, x_test,training=False):
        predictions = np.empty((0,1))
        batch_size=2
        for i in range(0,x_test.shape[0],batch_size):
            x_batch = x_test[i:i+batch_size]
            prediction_i =  self.sess.run(self.prediction,feed_dict={self.X:x_batch})
            predictions = np.concatenate((predictions,np.argmax(prediction_i,axis=1)[:,np.newaxis]))
        return predictions

    def get_accuracy(self,x_test,y_test,training=False):
        return self.sess.run(self.accuracy,feed_dict={self.X:x_test,self.Y:y_test})

    def train(self,x_train,y_train,training=True):
        return self.sess.run([self.train_op,self.increment_global_step],feed_dict={self.X: x_train,self.Y: y_train})

    def get_train_summary(self,acc):
        return self.sess.run(self.acc_summary,feed_dict={self.accuracy_ph:acc})

    def get_test_summary(self,acc):
        return self.sess.run(self.test_acc_summary,feed_dict={self.accuracy_ph:acc})

# Initialize the variables (i.e. assign their default value)
data_X,data_Y = load_dataset.main('/home/jumabek/data/wbc/new_5_crop_128_128')
data = np.concatenate((data_X,data_Y[:,np.newaxis]),axis=1)
np.random.seed(seed=SEED)

K=10
KFold_indices = []
stratifiedkfold = StratifiedKFold(n_splits=K,random_state=SEED,shuffle=True)
for train_index, test_index in stratifiedkfold.split(data_X, data_Y):
	KFold_indices.append((train_index,test_index))

#Y_test = convert_index_hotlabel(Y_test,num_classes)
	
best_test_acc_sum = 0
for k in range(K):
    fingerprint = '/lr_'+str(learning_rate) + '_bs_'+str(64)  + '_h_'+str(num_hidden)+'_s_' + str(validate_step) + '_ts_'+str(training_steps) + '_k_' +str(k)
    modeldir = './models' + fingerprint
    train_index, test_index = KFold_indices[k]
    X_train = data_X[train_index]
    Y_train = data_Y[train_index]
    Y_train = convert_index_hotlabel(Y_train,num_classes)
    X_test = data_X[test_index]
    Y_test = data_Y[test_index]
    Y_test = convert_index_hotlabel(Y_test,num_classes)
    
    train_size = Y_train.shape[0]
    test_size = Y_test.shape[0]
    X_train = X_train.reshape((train_size,timesteps,num_input))
    X_test = X_test.reshape((test_size,timesteps,num_input))

    # Start evaluating 
    best_model = None
    saved_path = None
    best_acc = 0

    tf.reset_default_graph()
    sess = tf.Session() 
    model = Model(sess,'fold-{}'.format(K))
    saver = tf.train.Saver() 
    saved_path = join(modeldir,"best.ckpt")
    saver.restore(sess,saved_path)
    loaded_step = sess.run(model.global_step)
    #sess.run(tf.global_variables_initializer())
    print("Fold-{}:\n".format(k))
    print('best step: ', loaded_step)
    predictions = model.predict_for_test(X_test)
    print(predictions.shape)
    correct_pred = (predictions==np.argmax(Y_test,axis=1))
    acc = np.mean(correct_pred)
    print(acc)
    #print(predictions[:5],Y_test[:5])
    continue
    
    train_acc = model.get_accuracy(X_train,Y_train)
    print("Train Accuracy:", train_acc)
    test_acc = model.get_accuracy(X_test,Y_test)
    print("Test Accuracy:", test_acc)

