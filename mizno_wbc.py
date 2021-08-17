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
training_data_dir = '/home/mizno/wbc/db/128_128_training/'
validation_data_dir = '/home/mizno/wbc/db/128_128_validation/'
test_data_dir = '/home/mizno/wbc/db/128_128_test/'
# '/home/mizno/wbc/db/128_128_training_validation/'

training_data_x, training_data_y, test_data_x, test_data_y = load_dataset.main(training_data_dir, validation_data_dir) #  training set, validation set or test set
print('mizno, training data x (image data) = ' + str(len(training_data_x)))
print('mizno, training data y (label) = ' + str(len(training_data_y)))
print('mizno, test data x (image data) = ' + str(len(test_data_x)))
print('mizno, test data y (label) = ' + str(len(test_data_y)))





###################################################
# build CNN model
class Model:

	def __init__(self, sess, name):
		self.sess = sess
		self.name = name
		self._build_net()

	def _build_net(self):
		with tf.variable_scope(self.name):
			# for dropout (keep_prob) rate
			self.training = tf.placeholder(tf.bool)
			print(str(self.training))

			# input place holders
			self.X = tf.placeholder(tf.float32, [None, 128 * 128 * 3]) #  '* 1' means grayscale, '* 3' means RGB
			print(str(self.X))
			self.Y = tf.placeholder(tf.float32, [None, n_classes])
			print(str(self.Y))
			X_img = tf.reshape(self.X, [-1, 128, 128, 3]) # for RGB, 1 means grayscale, 3 means RGB

			with tf.device('/gpu:0'):
				# Convolutional Layer #1 and Pooling Layer #1
				conv1 = tf.layers.conv2d(inputs=X_img, filters=16, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
				print('conv1:' + str(conv1))
				pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], padding="SAME", strides=2)
				print('pool1:' + str(pool1))
				dropout1 = tf.layers.dropout(inputs=pool1, rate=0.6, training=self.training)
				print('dropout1:' + str(dropout1))

			with tf.device('/gpu:1'):
				# Convolutional Layer #2 and Pooling Layer #2
				conv2 = tf.layers.conv2d(inputs=dropout1, filters=32, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu,  kernel_initializer=tf.contrib.layers.xavier_initializer())
				print('conv2:' + str(conv2))
				pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], padding="SAME", strides=2)
				print('pool2:' + str(pool2))
				dropout2 = tf.layers.dropout(inputs=pool2, rate=0.6, training=self.training)
				print('dropout2:' + str(dropout2))

			with tf.device('/gpu:0'):
				# Convolutional Layer #3 and Pooling Layer #3
				conv3 = tf.layers.conv2d(inputs=dropout2, filters=64, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu,  kernel_initializer=tf.contrib.layers.xavier_initializer())
				print('conv3:' + str(conv3))
				pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], padding="SAME", strides=2)
				print('pool3:' + str(pool3))
				dropout3 = tf.layers.dropout(inputs=pool3, rate=0.6, training=self.training)
				print('dropout3:' + str(dropout3))

			with tf.device('/gpu:1'):
				# Dense Layer with Relu
				flat = tf.reshape(dropout3, [-1, 16*16*64]) # flat should be same with size of last dropout
				print('flat:' + str(flat))
				dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)
				print('dense:' + str(dense))
				dropout_dense = tf.layers.dropout(inputs=dense, rate=0.6, training=self.training)
				print('dropout_dense:' + str(dropout_dense))

			self.logits = tf.layers.dense(inputs=dropout_dense, units=n_classes)
			print('logits:' + str(self.logits))

		# define cost/loss & optimizer
		self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
		print('cost:' + str(self.cost))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
		print('optimizer:' + str(self.optimizer))

		correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
		print('correct_prediction:' + str(correct_prediction))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		print('accuracy:' + str(self.accuracy))

	def predict(self, x_test, training=False):
		return self.sess.run(self.logits, feed_dict={self.X: x_test, self.training: training})

	def get_accuracy(self, x_test, y_test, training=False):
		return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.training: training})

	def train(self, x_data, y_data, training=True):
		return self.sess.run([self.cost, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data, self.training: training})





###################################################
# initialize
sess = tf.Session()
models = []
best_models = []
num_models = 5 # when 'OOM' or 'Dst tensor' error occurs, need to reduce num_models
for m in range(num_models):
	models.append(Model(sess, "model" + str(m)))

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






'''
####################################################
# Test model and check accuracy without BATCH for TRAINING SET and TEST SET
# OOM error may occur for the number of dataset

for m_idx, m in enumerate(models):
	print(m_idx, 'Training Set Accuracy:', m.get_accuracy(training_data_x, training_data_y))

test_size = len(test_data_y) # for ensemble
predictions = np.zeros([test_size, 5])
for m_idx, m in enumerate(models):
	print(m_idx, 'Test Set Accuracy:', m.get_accuracy(test_data_x, test_data_y))
	p = m.predict(test_data_x) # for ensemble
	predictions += p # for ensemble
'''





'''
####################################################
# Test model and check accuracy with BATCH for TRAINING SET
# 'atr' means acc of training
atr_batch_size = 10
atr_cur_idx = 0
atr_start = 0
atr_end = 0
atr_total_batch_size = int(len(training_data_x) / atr_batch_size)
atr_avg_list = np.zeros(len(models))

for atr_step in range(0, atr_total_batch_size):
	atr_start = atr_cur_idx
	atr_end = atr_cur_idx + atr_batch_size
	atr_batch_x = np.array(training_data_x[atr_start:atr_end])
	atr_batch_y = np.array(training_data_y[atr_start:atr_end])
	atr_cur_idx = atr_cur_idx + atr_batch_size
	# print('mizno, batch_x: ' + str(atr_batch_x))
	# print('mizno, batch_y: ' + str(atr_batch_y))
	# print('mizno, cur_batch_idx: ' + str(atr_cur_idx))

	for m_idx, m in enumerate(models):
		atr_result = m.get_accuracy(atr_batch_x, atr_batch_y)
		atr_avg_list[m_idx] += atr_result / atr_total_batch_size
		# print(m_idx, 'Training Set Accuracy:', atr_avg_list[m_idx])

for m_idx, m in enumerate(models):
	print(m_idx, 'Training Set Accuracy:', atr_avg_list[m_idx])
'''





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





'''
####################################################
# for ensemble, need 'test_size', 'predictions' and 'p' variables
# refer to 'Test model and check accuracy without BATCH for TRAINING SET and TEST SET'
# ensemble_correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(test_data_y, 1))
# ensemble_accuracy = tf.reduce_mean(tf.cast(ensemble_correct_prediction, tf.float32))
# print('Ensemble accuracy:', sess.run(ensemble_accuracy))
'''


