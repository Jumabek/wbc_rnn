import numpy as np
import os
import random
from scipy import misc
from sklearn import preprocessing

def def_read_images(dir):
	all_labels = [] # for def_read_images()
	img_arr = [] # for def_read_images()

	# print('mizno, ' + dir)
	for path, subdirs, files in os.walk(dir):
		files.sort()
		subdirs.sort()
		for filename in files:
			# print('mizno, ' + filename)
			file_path = os.path.join(path, filename)
			str_labels = os.path.basename(os.path.dirname(file_path))
			# str_labels means a directory such as a '01.NE', '02.ME' and so on
			# print('mizno, ' + str_labels) # ['01.NE', '02.ME' ...]
			all_labels.append(str(str_labels)) # input '01.NE', '02.ME' to array
			# all_labels is for directories of images
			img = misc.imread(file_path, flatten=False) # 'flatten = True' means grayscale, 'False' means RGB
			img = img.astype(float) # astype means type conversion to float
			img = np.asarray(img).reshape(-1) # -1 means variableness
			# print('mizno, ' + str(img)) # [239.844 240.203 238.964 ...] 
			img = preprocessing.scale(img) # scale means normalization (standardization?, regularization?)
			# print('mizno, ' + str(img)) # [0.89303072  0.12565155 -0.16949429 ...] 
			img_arr.append(img)
	# print('mizno, ' + str(all_labels)) # ['01.NE', '01.NE', ..., '02.ME', '02.ME', ...]
	# print('mizno, ' + str(len(all_labels)))
	return img_arr, all_labels

def def_convert_dirname_index(all_labels):
	labels_index = [] # for def_convert_dirname_index()
	# converting from 'directory's name' to index
	deduplicated_labels = list(set(all_labels)) # set : removing duplicate items
	n_classes = len(deduplicated_labels) # n_classes means 5 (category)
	deduplicated_labels.sort()
	# print('mizno, ' + str(deduplicated_labels)) # ['01.NE', '02.ME', ...]
	for i in all_labels:
		labels_index.append(deduplicated_labels.index(i))
	# print('mizno, ' + str(labels_index)) # [0, 0, 0, ..., 1, 1, 1, ...]
	return  labels_index, n_classes

def def_convert_index_hotlabel(labels_index, n_classes):
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
	return hot_labels

def def_make_dataset(img_arr, hot_labels):
	# making dataset(images and hot labels)
	dataset = [] # for def_make_dataset()
	for i in range(len(hot_labels)):
		dataset.append([img_arr[i], hot_labels[i]])
	# print('mizno, ' + dataset)
	return dataset

def def_check_dataset(images):
	# checking whether read dataset is ok or not
	print('mizno, the number of read data: '+str(len(dataset)))
	print('mizno, ' + str(len(images)-1))
	print('mizno, ' + img_arr[len(dataset)-1])
	print('mizno, ' + images[len(images)-1])

def def_load_training_and_test_images(directory):
	img_arr, all_labels = def_read_images(directory)
	labels_index, n_classes = def_convert_dirname_index(all_labels)
	hot_labels = def_convert_index_hotlabel(labels_index, n_classes)
	dataset = def_make_dataset(img_arr, hot_labels)
	return dataset

def main(training_dir, test_dir):
	# print('main function is called!!!')
	print('mizno, load data set')

	training_images = def_load_training_and_test_images(training_dir)
	training_images = np.array(training_images)
	# def_check_dataset(training_images)

	test_images = def_load_training_and_test_images(test_dir)
	test_images = np.array(test_images)
	# def_check_dataset(test_images)

	# print(len(training_images))
	# print(len(test_images))

	# At present, we divided a training set and a test set physically,
	# we load the sets respectively.
	# But if you want to load all image data at one time,
	# you should use 'random.shuffle' to divide a training set and a test set.
	# random.shuffle(all_image_set)
	# test_set_size = int(0.2 * len(all_image_set))

	training_data_x = list(training_images[:,0]) # image data
	training_data_y = list(training_images[:,1]) # label
	test_data_x = list(test_images[:,0]) # image data
	test_data_y = list(test_images[:,1]) # label

	print('mizno, training data x (image data) = ' + str(len(training_data_x)))
	print('mizno, training data y (label) = ' + str(len(training_data_y)))
	print('mizno, test data x (image data) = ' + str(len(test_data_x)))
	print('mizno, test data y (label) = ' + str(len(test_data_y)))
	print('mizno, all data set has ' + str(len(training_images[0])) + ' columns (images data, label)')

	return training_data_x, training_data_y, test_data_x, test_data_y

if __name__ == '__main__':
	training_dir = 'training/' # mizno ??
	test_dir = 'test/' # mizno ??
	main(training_dir, test_dir)

