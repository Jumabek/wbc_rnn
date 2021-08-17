from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import load_dataset
import numpy as np
from os.path import join

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
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''
RESUME= False
SEED=123
# Training Parameters
learning_rate = 0.3 
training_steps = 100000
batch_size = 64
validate_step = 50

# Network Parameters
num_input = 128*3 # MNIST data input (img shape: 28*28)
timesteps = 128 # timesteps
num_hidden = 16 # hidden layer num of features
num_classes = 5 # MNIST total classes (0-9 digits)

logdir = './log'
fingerprint = '/lr_'+str(learning_rate) + '_bs_'+str(batch_size)  + '_h_'+str(num_hidden)+'_s_' + str(validate_step) + '_ts_'+str(training_steps)
modeldir = './models' + fingerprint

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    #lstm_cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell');
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
global_step = tf.Variable(0, trainable=False, name='global_step')
increment_global_step = tf.assign_add(global_step,1,
                                            name = 'increment_global_step')

train_op = optimizer.minimize(loss_op, global_step=global_step)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

accuracy_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
loss_ph = tf.placeholder(tf.float32,shape=None, name='loss_summary')

acc_summary = tf.summary.scalar('train_accuracy',accuracy_ph)
loss_summary = tf.summary.scalar('loss', loss_ph)
performance_summaries = tf.summary.merge([acc_summary,loss_summary])

val_acc_summary = tf.summary.scalar('val_accuracy',accuracy_ph)
test_acc_summary = tf.summary.scalar('test_accuracy',accuracy_ph)

writer = tf.summary.FileWriter(logdir+fingerprint)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
data_X,data_Y = load_dataset.main('/home/jumabek/data/wbc/new_5_crop_128_128')
data = np.concatenate((data_X,data_Y[:,np.newaxis]),axis=1)
np.random.seed(seed=SEED)
print(data.shape)
np.random.shuffle(data)
print(data.shape)

data = tf.convert_to_tensor(data,dtype=tf.float32)
data_train, data_test,data_val = tf.split(data,[5504,512,546],0)
with tf.Session() as sess:
	X_train = data_train[:,:-1].eval()
	Y_train = data_train[:,-1:].eval()
	Y_train = convert_index_hotlabel(Y_train,num_classes)
	
	X_val = data_val[:,:-1].eval()
	Y_val = data_val[:,-1:].eval()
	Y_val = convert_index_hotlabel(Y_val,num_classes)
	
	X_test = data_test[:,:-1].eval()
	Y_test = data_test[:,-1:].eval()
	Y_test = convert_index_hotlabel(Y_test,num_classes)
	

train_size = Y_train.shape[0]
saver = tf.train.Saver()
# Start training
best_model = None
saved_path = None
best_acc = 0
with tf.Session() as sess:

    # Run the initializer
    if RESUME:
        saved_path = join(modeldir,'best.ckpt') 
        saver.restore(sess,saved_path)
        loaded_step = sess.run(global_step)
    else:
        loaded_step = 1   
        sess.run(init)

    for step in range(max(1,loaded_step), training_steps+1):
        start = ((step-1)*batch_size)%train_size
        end = (step*batch_size)%train_size
        if start>=end:
            continue
        
        batch_x = X_train[start:end]
        batch_y = Y_train[start:end]
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = np.reshape(batch_x,(batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run([train_op,increment_global_step], feed_dict={X: batch_x, Y: batch_y})
        if step % validate_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            #                                                    Y:np.ones((128,5))})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            summ = sess.run(performance_summaries,feed_dict={accuracy_ph:acc,loss_ph:loss})
            writer.add_summary(summ,step)

            #validate
            val_num = Y_val.shape[0]
            X_val =  np.reshape(X_val,(val_num,timesteps,num_input))
            acc = sess.run(accuracy, feed_dict={X:X_val,Y:Y_val})
            if acc>=best_acc:
                saved_path = saver.save(sess,join(modeldir,'best.ckpt'))

            summ = sess.run(val_acc_summary,feed_dict={accuracy_ph:acc})
            writer.add_summary(summ,step)

    print("Optimization Finished!")

with tf.Session() as sess:
    saver.restore(sess,saved_path)
    # Calculate accuracy for 128 mnist test images
    test_num = Y_test.shape[0]
    X_test = np.reshape(X_test,(test_num,timesteps,num_input))
    test_acc = sess.run(accuracy,feed_dict={X:X_test,Y:Y_test})
    print("Testing Accuracy:", test_acc)

    summ = sess.run(test_acc_summary,feed_dict={accuracy_ph:test_acc})
    writer.add_summary(summ,1)
    writer.close()    

