# juma copied mizno's code and then inserted/modified VGGnet from 
#https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19_trainable.py
###################################################
# memo
# mizno, to do list
# Parameter 1: set_random_seed
# Parameter 2: filters, kernel_size, pool_size, strides, rate (dropout)
# Parameter 3: learning_rate, training_epochs
# Parameter 4: # of Hidden Layer, # of Units
# To increase # of layer easily, need to change code structure using like this 'n_filters = [32, 64]'
# Automatically change to better model(Choose better model from TRAINING and RESTORE)
# Batch Normalization, Leaky ReLU





###################################################
# Code starts
import tensorflow as tf
import numpy as np
import load_dataset
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
m_seed = 777 # 777, 776, 770, ...
tf.set_random_seed(m_seed) # reproducibility, default 777
np.random.seed(m_seed) # reproducibility, default 777
print('tf.set_random_seed: ' + str(m_seed))
print('np.random.seed: ' + str(m_seed))
n_classes = 5 # 01_BA, 02_EO, 03_LY, 04_MO, 05_NE

# to check dimension and OOM errors
# training_epochs = 1
# batch_size = 1
# num_models = 1

# hyper parameters
learning_rate = 0.0001
training_epochs = 300
batch_size = 5 # when OOM error occurs, need to reduce batch_size
print('learning_rate: ' + str(learning_rate))
print('training_epochs: ' + str(training_epochs))
print('batch_size: ' + str(batch_size))





###################################################
# load images, you can select 128*128 or 256*256
training_data_dir = '/home/mizno/wbc/db/new_5_crop_224_224/'
validation_data_dir = '/home/mizno/wbc/db/new_5_crop_224_224/'
test_data_dir = '/home/mizno/wbc/db/new_5_224_224/'
# '/home/mizno/wbc/db/128_128_training_validation/'

training_data_x, training_data_y, test_data_x, test_data_y = load_dataset.main(training_data_dir, validation_data_dir) #  training set, validation set or test set
print('mizno, training data x (image data) = ' + str(len(training_data_x)))
print('mizno, training data y (label) = ' + str(len(training_data_y)))
print('mizno, test data x (image data) = ' + str(len(test_data_x)))
print('mizno, test data y (label) = ' + str(len(test_data_y)))



###################################################
# build CNN model
class Model:

    def __init__(self, sess, name, num_class = 5, dropout=0.5,mode='train',device='/gpu:0'):
        self.sess = sess
        self.name = name
        self.num_class = num_class
        self.mode = mode
        self.var_dict = {}
        self.dropout = dropout
        self.data_dict = None
        self._build_net(device)


    def _build_net(self,device):
        with tf.variable_scope(self.name):
            # for dropout (keep_prob) rate
            start_time = time.time()
            self.training = tf.placeholder(tf.bool)
            print(str(self.training))
            # input place holders
            self.X = tf.placeholder(tf.float32, [None, 224 * 224 * 3]) #  '* 1' means grayscale, '* 3' means RGB
            print(str(self.X))
            self.Y = tf.placeholder(tf.float32, [None, n_classes])
            print(str(self.Y))
            X_img = tf.reshape(self.X, [-1, 224, 224, 3]) # for RGB, 1 means grayscale, 3 means RGB

            if 1==1:
                # Convolutional Layer #1 and Pooling Layer #1
                self.conv1_1 = self.conv_layer(X_img,3,64, "conv1_1")
                self.conv1_2 = self.conv_layer(self.conv1_1,64,64, "conv1_2")
                self.pool1 = self.max_pool(self.conv1_2, 'pool1')

                self.conv2_1 = self.conv_layer(self.pool1, 64,128, "conv2_1")
                self.conv2_2 = self.conv_layer(self.conv2_1,128,128, "conv2_2")
                self.pool2 = self.max_pool(self.conv2_2, 'pool2')

                self.conv3_1 = self.conv_layer(self.pool2,128, 256, "conv3_1")
                self.conv3_2 = self.conv_layer(self.conv3_1,256, 256, "conv3_2")
                self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
                self.pool3 = self.max_pool(self.conv3_3, 'pool3')

                self.conv4_1 = self.conv_layer(self.pool3,256,512, "conv4_1")
                self.conv4_2 = self.conv_layer(self.conv4_1,512,512, "conv4_2")
                self.conv4_3 = self.conv_layer(self.conv4_2,512,512, "conv4_3")
                self.pool4 = self.max_pool(self.conv4_3, 'pool4')

                self.conv5_1 = self.conv_layer(self.pool4,512, 512, "conv5_1")
                self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
                self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
                self.pool5 = self.max_pool(self.conv5_3, 'pool5')
                
                self.fc6 = self.fc_layer(self.pool5,25088,4096, "fc6") # ((224//(2**5))**2)* 512
                self.relu6 = tf.nn.relu(self.fc6)
                if self.mode=='train':
                    self.relu6 =  tf.nn.dropout(self.relu6, self.dropout)
                self.fc7 = self.fc_layer(self.relu6, 4096, 1024,"fc7")
                self.relu7 = tf.nn.relu(self.fc7)
                
                if self.mode=='train':
                    self.relu7 = tf.nn.dropout(self.relu7,self.dropout)

                self.fc8 = self.fc_layer(self.relu7, 1024,self.num_class, "fc8")
                
                self.prob = tf.nn.softmax(self.fc8, name="prob")

                self.data_dict = None
                print(("build model finished: %ds" % (time.time() - start_time)))

            # define cost/loss & optimizer
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.fc8, labels=self.Y))
            print('cost:' + str(self.cost))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
            print('optimizer:' + str(self.optimizer))

            correct_prediction = tf.equal(tf.argmax(self.fc8, 1), tf.argmax(self.Y, 1))
            print('correct_prediction:' + str(correct_prediction))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print('accuracy:' + str(self.accuracy))

    def avg_pool(self, bottom, name):
                return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
                return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
                with tf.variable_scope(name):
                    filt, conv_biases = self.get_conv_var(3,in_channels,out_channels,name)

                    conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

                    bias = tf.nn.bias_add(conv, conv_biases)
                    relu = tf.nn.relu(bias)

                    return relu

    def fc_layer(self, bottom, in_size, out_size, name):
                with tf.variable_scope(name):
                    weights, biases = self.get_fc_var(in_size,out_size,name)

                    x = tf.reshape(bottom,[-1, in_size])
                    fc = tf.nn.bias_add(tf.matmul(x,weights), biases)
                    
                    return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
                initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels],0.0,0.001)
                filters = self.get_var(initial_value, name, 0, name +'_filters')

                initial_value = tf.truncated_normal([out_channels], .0, .001)
                biases = self.get_var(initial_value, name, 1, name+ '_biases')

                return filters, biases

    def get_fc_var(self, in_size, out_size,name):
                initial_value = tf.truncated_normal([in_size, out_size],.0, .001)
                weights = self.get_var(initial_value, name, 1, name +'_weights')

                initial_value = tf.truncated_normal([out_size], .0, .001)
                biases = self.get_var(initial_value, name, 1, name+ '_biases')
                return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
                
        	if self.data_dict is not None and name in self.data_dict:
            		value = self.data_dict[name][idx]
        	else:
            		value = initial_value

        	if self.mode=='train':
            		var = tf.Variable(value, name=var_name)
        	else:
            		var = tf.constant(value, dtype=tf.float32, name=var_name)

        	self.var_dict[(name, idx)] = var

        	# print var_name, var.get_shape().as_list()
        	assert var.get_shape() == initial_value.get_shape()

        	return var

        #Juma says: you may not need this function, but adding it in case you would like to save it
    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
        	assert isinstance(sess, tf.Session)

        	data_dict = {}

        	for (name, idx), var in list(self.var_dict.items()):
            		var_out = sess.run(var)
            		if name not in data_dict:
                		data_dict[name] = {}
            		data_dict[name][idx] = var_out

        	np.save(npy_path, data_dict)
        	print(("file saved", npy_path))
        	return npy_path

	
    def get_var_count(self):
        	count = 0
        	for v in list(self.var_dict.values()):
            		count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        	return count





    def predict(self, x_test, training=False):
        return self.sess.run(self.fc8, feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})





###################################################
# initialize
sess = tf.Session()
models = []
best_models = []
num_models = 1 # when 'OOM' or 'Dst tensor' error occurs, need to reduce num_models
for m in range(num_models):
    models.append(Model(sess, "model" + str(m),num_class=n_classes,device="/device:GPU:1"))

sess.run(tf.global_variables_initializer())






####################################################
# train my model with BATCH
print('Learning Started!')
for epoch in range(training_epochs):
	avg_cost_list = np.zeros(len(models))
	total_batch = int(len(training_data_x) / batch_size)
	# print('mizno, total_batch: ' + str(total_batch))
	cur_batch_idx = 0

	for i in range(0, total_batch):
		start = cur_batch_idx
		end = cur_batch_idx + batch_size
		batch_x = np.array(training_data_x[start:end])
		batch_y = np.array(training_data_y[start:end])
		cur_batch_idx = cur_batch_idx + batch_size
		# print('mizno, batch_x: ' + str(batch_x))
		# print('mizno, batch_y: ' + str(batch_y))
		# print('mizno, cur_batch_idx: ' + str(cur_batch_idx))

		# train each model
		for m_idx, m in enumerate(models):
			c, _ = m.train(batch_x, batch_y) # train function returns cost and optimizer
			avg_cost_list[m_idx] += c / total_batch

	print('Epoch:', '%04d' % (epoch + 1), 'cost =', avg_cost_list)

print('Learning Finished!')






####################################################
# Test model and check accuracy with BATCH for TEST SET
acc_batch_size = 10
acc_cur_idx = 0
acc_start = 0
acc_end = 0
acc_total_batch_size = int(len(test_data_x) / acc_batch_size)
acc_avg_list = np.zeros(len(models))

saver = tf.train.Saver()

for acc_step in range(0, acc_total_batch_size):
	acc_start = acc_cur_idx
	acc_end = acc_cur_idx + acc_batch_size
	acc_batch_x = np.array(test_data_x[acc_start:acc_end])
	acc_batch_y = np.array(test_data_y[acc_start:acc_end])
	acc_cur_idx = acc_cur_idx + acc_batch_size
	# print('mizno, batch_x: ' + str(acc_batch_x))
	# print('mizno, batch_y: ' + str(acc_batch_y))
	# print('mizno, cur_batch_idx: ' + str(acc_cur_idx))

	for m_idx, m in enumerate(models):
		acc_result = m.get_accuracy(acc_batch_x, acc_batch_y)
		acc_avg_list[m_idx] += acc_result / acc_total_batch_size
		# print(m_idx, 'Test Set Accuracy:', acc_avg_list[m_idx])

for m_idx, m in enumerate(models):
	best_models.append(acc_avg_list[m_idx])
	print(m_idx, 'Test Set Accuracy:', acc_avg_list[m_idx])
	# save all models
	saver.save(models[m_idx].sess, "./model/resources/model00" + str(m_idx) + "/model")
	print("saved model = " + str(m_idx))





####################################################
# save a best model
# best_model = models[np.argmax(best_models)]
# saver = tf.train.Saver()
# saver.save(best_model.sess, "./model/model")
# print(np.argmax(best_models))



