# CS231N Final Project
# Henry Lin (henryln1@stanford.edu) and Christopher Vo (cvo9@stanford.edu)
# Implementation of initial VGGNet-inspired model (scaled down)


# Import needed libraries and functions
import tensorflow as tf
import numpy as np
from load_data import *
from data_batch import Data
from random import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import os
import time


#device = '/cpu:0'

device = '/gpu:0'

resize_height, resize_width = 216, 384
image_set_size = 12
FR = image_set_size
W = resize_height
H = resize_width
D = 3
num_classes = 10

#device = '/gpu:0'

#device = "/device:GPU:0"

# TODO: load functions that read and process input data
# Inputs: 
#	path: name of path to data
# Returns:
#	x_train, y_train, x_val, y_val, x_test, y_test
# Image frame sets have shape (FR, W, H, D) where FR = # frames per image set, also known as depth in TF (10), W = width, H = height, D = color dimensions, also known as channels in TF (3)
# Number of classes: C = 10
def load_data(path):
	train_path_image_dict = read_all_images("train")
	train_Data = Data(train_path_image_dict)
	validation_path_image_dict = read_all_images("validation")
	val_Data = Data(validation_path_image_dict)

	x_train, y_train = train_Data.create_batch()
	x_val, y_val = val_Data.create_batch()
	x_test = None
	y_test = None
	print("shape of x train: ", x_train.shape)
	print("shape of y_train: ", y_train.shape)
	print("shape of x val: ", x_val.shape)
	print("shape of y_val: ", y_val.shape)
	return x_train, y_train, x_val, y_val, x_test, y_test

def load_data_new(path):
	train_path_image_dict = read_all_images("train")
	train_Data = Data(train_path_image_dict)
	validation_path_image_dict = read_all_images("validation")
	val_Data = Data(validation_path_image_dict)

	return train_Data, val_Data


# Function that initializes the VGGNet-inspired model
# Inputs:
#	input_shape: shape of each input image set, (Fr, W, H, D) in this case
# Returns:
#	vgg_model: compiled model ready to be trained
# Architecture:
#	Conv layers: 3x3 filter size, stride 1, padding = 'same', with ReLU
#	Pool layers: 2x2 max pool, stride 2
#	Structure: 2 Conv (16 filters), Pool, 2 Conv (32 filters), Pool, 3 Conv (64 filters), Pool, FC
#	TODO: Should incoporate Batch Norm + Dropout
#	Loss: softmax
def vgg_model_init(inputs):

	# Define hyperparams, weight initializer, activation, regularization, loss function, and optimizer
	#FR, W, H, D = input_shape
	#resize_height, resize_width = 21, 256
	image_set_size = 8
	FR = image_set_size
	W = resize_height
	H = resize_width
	D = 3
	num_classes = 10

	input_shape = (FR, W, H, D)

	num_filters = [64, 128, 256] # number of filters in each set of conv layers
	filter_size = 3 # 3x3 filters
	filter_stride = 1

	pool_size = 2 # 2x2 max pool
	pool_stride = 2

	initializer = tf.variance_scaling_initializer(scale=2.0) # initializer for weights
	activation = tf.nn.relu # ReLU for each Conv layer

	reg_strength = 0.1
	regularization = tf.contrib.layers.l2_regularizer(reg_strength) # L2 regularization for FC layer

	#loss = tf.nn..ftmax_cross_entropy_with_logits # NOTE: not sure if this loss function works with the Keras compile/fit model methods, may have to manually implement train method like in homework

	#optimizer = tf.train.AdamOptimizer() # optimizer used is Adam

	print("num classes:", num_classes)


	#LSTM VERSION 

	lstm_layer_output_size = 300
	layers = [

		# Conv Layer Set 1: 2 Conv layers (16 filters), 1 Pool layer
		tf.layers.Conv3D(input_shape=input_shape, filters=num_filters[0], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.Conv3D(filters=num_filters[0], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		#tf.layers.BatchNormalization(),
		tf.layers.MaxPooling3D(pool_size=[1, pool_size, pool_size], strides=[1, pool_stride, pool_stride], padding='valid'),
		# # Conv Layer Set 2: 2 Conv layers (32 filters), 1 Pool layer
		tf.layers.Conv3D(filters=num_filters[1], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.Conv3D(filters=num_filters[1], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		#tf.layers.BatchNormalization(),
		tf.layers.MaxPooling3D(pool_size=[1, pool_size, pool_size], strides=[1, pool_stride, pool_stride], padding='valid'),

		# # Conv Layer Set 3: 3 Conv layers (64 filters), 1 Pool layer
		tf.layers.Conv3D(filters=num_filters[2], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.Conv3D(filters=num_filters[2], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.Conv3D(filters=num_filters[2], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		#tf.layers.BatchNormalization(),		
		tf.layers.MaxPooling3D(pool_size=[1, pool_size, pool_size], strides=[1, pool_stride, pool_stride], padding='valid'),

		#conv layer set 4:
		# tf.layers.Conv3D(filters=num_filters[2], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		# tf.layers.Conv3D(filters=num_filters[2], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		# tf.layers.Conv3D(filters=num_filters[2], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		# tf.layers.MaxPooling3D(pool_size=[1, pool_size, pool_size], strides=[1, pool_stride, pool_stride], padding='valid'),

		# FC Layer
		#tf.layers.Flatten(shape = ()),
		#tf.layers.Dense(units = num_classes, kernel_initializer=initializer, kernel_regularizer=regularization)
		tf.keras.layers.Reshape((image_set_size, 18 * 32 * 256)),
		#tf.keras.layers.LSTM(units = image_set_size * 18 * 32 * 256),
		#tf.keras.layers.Reshape((image_set_size, 18 * 32 * 256)),
		tf.keras.layers.LSTM(units = num_classes),
		#tf.layers.Dense(units = num_classes, kernel_initializer=initializer, kernel_regularizer=regularization)
	]



	# Initialize model and compile
	vgg_model = tf.keras.Sequential(layers)
	#vgg_model.compile(loss='mean_squared_error', optimizer=optimizer)

	#return vgg_model
	return vgg_model(inputs)


def vgg_model_single_image_init(inputs):

	input_shape = (W, H, D)

	num_filters = [64, 128, 256] # number of filters in each set of conv layers
	filter_size = 3 # 3x3 filters
	filter_stride = 1

	pool_size = 2 # 2x2 max pool
	pool_stride = 2

	initializer = tf.variance_scaling_initializer(scale=2.0) # initializer for weights
	activation = tf.nn.relu # ReLU for each Conv layer

	reg_strength = 0.1
	regularization = tf.contrib.layers.l2_regularizer(reg_strength) # L2 regularization for FC layer

	dense_layer_unit_count = 128
	layers = [
	tf.keras.layers.Conv2D(input_shape = input_shape, filters = num_filters[0], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer),
	tf.keras.layers.Conv2D(filters = num_filters[0], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer),
	tf.keras.layers.MaxPooling2D(pool_size = pool_size, strides = pool_stride, padding = 'valid'),


	tf.keras.layers.Conv2D(filters = num_filters[1], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer),
	tf.keras.layers.Conv2D(filters = num_filters[1], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer),
	#tf.keras.layers.Dropout(rate = 0.1),
	tf.keras.layers.MaxPooling2D(pool_size = pool_size, strides = pool_stride, padding = 'valid'),


	tf.keras.layers.Conv2D(filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer),
	tf.keras.layers.Conv2D(filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer),
	tf.keras.layers.Conv2D(filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer),
	#tf.keras.layers.Dropout(rate = 0.1),
	tf.keras.layers.MaxPooling2D(pool_size = pool_size, strides = pool_stride, padding = 'valid'),
	tf.layers.Flatten(),
	#tf.keras.layers.Dense(units = 32, kernel_initializer=initializer, kernel_regularizer=regularization),
	tf.keras.layers.Dense(units = num_classes, kernel_initializer=initializer, kernel_regularizer=regularization),
	#tf.keras.layers.Dropout(rate = 0.5)
	]

	vgg_model = tf.keras.Sequential(layers)

	return vgg_model(inputs)


def vgg_model_single_image_init_non_seq(inputs):
	#resize_height, resize_width = 144, 256
	#image_set_size = 8
	#FR = image_set_size
	W = resize_height
	H = resize_width
	D = 3
	num_classes = 10

	input_shape = (W, H, D)

	num_filters = [64, 128, 256] # number of filters in each set of conv layers
	filter_size = 3 # 3x3 filters
	filter_stride = 1

	pool_size = 2 # 2x2 max pool
	pool_stride = 2

	initializer = tf.variance_scaling_initializer(scale=2.0) # initializer for weights
	activation = tf.nn.relu # ReLU for each Conv layer

	reg_strength = 0.1
	regularization = tf.contrib.layers.l2_regularizer(reg_strength) # L2 regularization for FC layer

	conv1 = tf.layers.conv2d(inputs, filters = num_filters[0], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn1 = tf.layers.batch_normalization(conv1, training=True)
	conv2 = tf.layers.conv2d(bn1, filters = num_filters[0], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn2 = tf.layers.batch_normalization(conv2, training=True)
	pool1 = tf.layers.max_pooling2d(bn2, pool_size = pool_size, strides = pool_stride, padding = 'valid')
	drop1 = tf.layers.dropout(pool1, rate=0.2)

	conv3 = tf.layers.conv2d(drop1, filters = num_filters[1], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn3 = tf.layers.batch_normalization(conv3, training=True)
	conv4 = tf.layers.conv2d(bn3, filters = num_filters[1], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn4 = tf.layers.batch_normalization(conv4, training=True)
	pool2 = tf.layers.max_pooling2d(bn4, pool_size = pool_size, strides = pool_stride, padding = 'valid')
	drop2 = tf.layers.dropout(pool2, rate=0.2)

	conv5 = tf.layers.conv2d(drop2, filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn5 = tf.layers.batch_normalization(conv5, training=True)
	conv6 = tf.layers.conv2d(bn5, filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn6 = tf.layers.batch_normalization(conv6, training=True)
	conv7 = tf.layers.conv2d(bn6, filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn7 = tf.layers.batch_normalization(conv7, training=True)
	pool3 = tf.layers.max_pooling2d(bn7, pool_size = pool_size, strides = pool_stride, padding = 'valid')
	drop3 = tf.layers.dropout(pool3, rate=0.2)
	
	flat1 = tf.layers.flatten(drop3)
	output = tf.layers.dense(flat1, units = num_classes, kernel_initializer=initializer, kernel_regularizer=regularization)

	return output



def vgg_model_single_image_init_non_seq_lstm(inputs):

	'''
	inputs is a 5-D tensor that we must shape into 4-D before convolutions
	rehshape back into 5-D and then into 3-D for lstm layer

	'''
	#resize_height, resize_width = 144, 256
	#image_set_size = 8
	#FR = image_set_size
	W = resize_height
	H = resize_width
	D = 3
	num_classes = 10
	#batch_size = inputs.get_shape()[0]
	input_shape = (W, H, D)

	num_filters = [64, 128, 256] # number of filters in each set of conv layers
	filter_size = 3 # 3x3 filters
	filter_stride = 1

	pool_size = 2 # 2x2 max pool
	pool_stride = 2

	initializer = tf.variance_scaling_initializer(scale=2.0) # initializer for weights
	activation = tf.nn.relu # ReLU for each Conv layer

	reg_strength = 0.1
	regularization = tf.contrib.layers.l2_regularizer(reg_strength) # L2 regularization for FC layer

	#print("input shape: ", inputs.get_shape())
	inputs = tf.reshape(inputs, (-1, W, H, D))
	#print("input reshaped: ", tf.shape(inputs))
	conv1 = tf.layers.conv2d(inputs, filters = num_filters[0], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn1 = tf.layers.batch_normalization(conv1, training=True)
	conv2 = tf.layers.conv2d(bn1, filters = num_filters[0], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn2 = tf.layers.batch_normalization(conv2, training=True)
	pool1 = tf.layers.max_pooling2d(bn2, pool_size = pool_size, strides = pool_stride, padding = 'valid')
	drop1 = tf.layers.dropout(pool1, rate=0.2)

	conv3 = tf.layers.conv2d(drop1, filters = num_filters[1], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn3 = tf.layers.batch_normalization(conv3, training=True)
	conv4 = tf.layers.conv2d(bn3, filters = num_filters[1], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn4 = tf.layers.batch_normalization(conv4, training=True)
	pool2 = tf.layers.max_pooling2d(bn4, pool_size = pool_size, strides = pool_stride, padding = 'valid')
	drop2 = tf.layers.dropout(pool2, rate=0.2)

	conv5 = tf.layers.conv2d(drop2, filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn5 = tf.layers.batch_normalization(conv5, training=True)
	conv6 = tf.layers.conv2d(bn5, filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn6 = tf.layers.batch_normalization(conv6, training=True)
	conv7 = tf.layers.conv2d(bn6, filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn7 = tf.layers.batch_normalization(conv7, training=True)
	pool3 = tf.layers.max_pooling2d(bn7, pool_size = pool_size, strides = pool_stride, padding = 'valid')
	drop3 = tf.layers.dropout(pool3, rate=0.2)
	
	# print("drop3 shape: ", tf.shape(drop3))
	# reshaped = tf.reshape(drop3, (-1, image_set_size, W, H, D))
	# print("reshaped shape: ", tf.shape(reshaped))
	# flattened = tf.reshape(reshaped,(-1, image_set_size, -1))
	# print("flattened shape: ", tf.shape(flattened))
	# lstm_cell = tf.contrib.rnn.LSTMCell(num_units = 10)

	# outputs, states = tf.nn.dynamic_rnn(lstm_cell, flattened, dtype = tf.float32)

	# output = outputs[-1]
	# print("shape of output: ", tf.shape(output))

	#flat1 = tf.layers.flatten(drop3)
	#output = tf.layers.dense(flat1, units = num_classes, kernel_initializer=initializer, kernel_regularizer=regularization)
	#print("shape of drop3: ", drop3)
	layers = [
		#tf.keras.layers.Reshape((-1, image_set_size, 18, 32, 256)),
		tf.keras.layers.Reshape((image_set_size, 18 * 32 * 256)),
		tf.keras.layers.LSTM(units = num_classes)
	]
	vgg_model = tf.keras.Sequential(layers)
	return vgg_model(drop3)
	return output


def vgg_model_conv3d_init(inputs):

	'''
	inputs is a 5-D tensor that we must shape into 4-D before convolutions
	rehshape back into 5-D and then into 3-D for lstm layer

	'''
	#resize_height, resize_width = 144, 256
	#image_set_size = 8
	#FR = image_set_size
	W = resize_height
	H = resize_width
	D = 3
	num_classes = 10
	#batch_size = inputs.get_shape()[0]
	input_shape = (FR, W, H, D)

	num_filters = [64, 128, 256] # number of filters in each set of conv layers
	filter_size = 3 # 3x3 filters
	filter_stride = 1

	pool_size = 2 # 2x2 max pool
	pool_stride = 2

	initializer = tf.variance_scaling_initializer(scale=2.0) # initializer for weights
	activation = tf.nn.relu # ReLU for each Conv layer

	reg_strength = 0.2
	regularization = tf.contrib.layers.l2_regularizer(reg_strength) # L2 regularization for FC layer

	#print("input shape: ", inputs.get_shape())
	#inputs = tf.reshape(inputs, (-1, W, H, D))
	#print("input reshaped: ", tf.shape(inputs))
	conv1 = tf.layers.conv3d(inputs, filters = num_filters[0], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn1 = tf.layers.batch_normalization(conv1, training=True)
	conv2 = tf.layers.conv3d(bn1, filters = num_filters[0], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn2 = tf.layers.batch_normalization(conv2, training=True)
	pool1 = tf.layers.max_pooling3d(bn2, pool_size = (1, pool_size, pool_size), strides = (1, pool_stride, pool_stride), padding = 'valid')
	drop1 = tf.layers.dropout(pool1, rate=0.3)

	conv3 = tf.layers.conv3d(drop1, filters = num_filters[1], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn3 = tf.layers.batch_normalization(conv3, training=True)
	conv4 = tf.layers.conv3d(bn3, filters = num_filters[1], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn4 = tf.layers.batch_normalization(conv4, training=True)
	pool2 = tf.layers.max_pooling3d(bn4, pool_size = (1, pool_size, pool_size), strides = (1, pool_stride, pool_stride), padding = 'valid')
	drop2 = tf.layers.dropout(pool2, rate=0.3)

	conv5 = tf.layers.conv3d(drop2, filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn5 = tf.layers.batch_normalization(conv5, training=True)
	conv6 = tf.layers.conv3d(bn5, filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn6 = tf.layers.batch_normalization(conv6, training=True)
	conv7 = tf.layers.conv3d(bn6, filters = num_filters[2], kernel_size = filter_size, strides = filter_stride, padding = 'same', activation = activation, kernel_initializer = initializer)
	bn7 = tf.layers.batch_normalization(conv7, training=True)
	pool3 = tf.layers.max_pooling3d(bn7, pool_size = (1, pool_size, pool_size), strides = (1, pool_stride, pool_stride), padding = 'valid')
	drop3 = tf.layers.dropout(pool3, rate=0.3)
	
	# print("drop3 shape: ", tf.shape(drop3))
	# reshaped = tf.reshape(drop3, (-1, image_set_size, W, H, D))
	# print("reshaped shape: ", tf.shape(reshaped))
	# flattened = tf.reshape(reshaped,(-1, image_set_size, -1))
	# print("flattened shape: ", tf.shape(flattened))
	# lstm_cell = tf.contrib.rnn.LSTMCell(num_units = 10)

	# outputs, states = tf.nn.dynamic_rnn(lstm_cell, flattened, dtype = tf.float32)

	# output = outputs[-1]
	# print("shape of output: ", tf.shape(output))

	flat1 = tf.layers.flatten(drop3)
	output = tf.layers.dense(flat1, units = num_classes, kernel_initializer=initializer, kernel_regularizer=regularization)
	#print("shape of drop3: ", drop3)
	# layers = [
	# 	#tf.keras.layers.Reshape((-1, image_set_size, 18, 32, 256)),
	# 	tf.keras.layers.Reshape((image_set_size, 18 * 32 * 256)),
	# 	tf.keras.layers.LSTM(units = num_classes)
	# ]
	# vgg_model = tf.keras.Sequential(layers)
	# return vgg_model(drop3)
	return output




# def train_part34_single_image_non_seq():

# 	x = tf.placeholder(tf.float32, [None, W, H, D])
# 	y = tf.placeholder(tf.int32, [None])

# 	scores = vgg_model_single_image_init_non_seq(x)

# 	optimizer = optimizer_init_fn()

# 	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
# 	loss = tf.reduce_mean(loss)

# 	#regularization
# 	reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# 	loss += regularization_strength * sum(reg_losses)

# 	train_step = optimizer.minimize(loss)
# 	extra_step = tf.get_collection(tf.GraphKeys.UPDATE_OPS, 'vgg')


def check_accuracy_single_frame(sess, x, scores, dataset = 'validation', is_training = None):

	batch_size = 50
	number_batches_to_check = 10
	num_correct, num_samples = 0, 0

	all_y_pred = []
	all_y_actual = []
	for i in range(number_batches_to_check):
		#x_batch, y_batch = load_single_frame_batch(batch_size, dataset = dataset)
		
		x_batch, y_batch = load_batch_multiple_frames_into_single(batch_size, dataset = dataset)
		feed_dict = {x: x_batch, is_training: 0}
		scores_np = sess.run(scores, feed_dict=feed_dict)
		y_pred = scores_np.argmax(axis=1)
		num_samples += x_batch.shape[0]
		num_correct += (y_pred == y_batch).sum()

		all_y_pred += y_pred.tolist()
		all_y_actual += y_batch.tolist()

	#F1_score = f1_score(all_y_actual, all_y_pred, average = 'micro')
	precision, recall, F1_score, support = precision_recall_fscore_support(all_y_actual, all_y_pred, average='micro')


	acc = num_correct / num_samples
	print("F1 Score: ", F1_score)
	print("Precision: ", precision)
	print("Recall: ", recall)
	print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

	return



def train_part34_single_image(model_init_fn, optimizer_init_fn, num_epochs=10):


	model_run_name = 'single_frames_model_batch_128_reg_strength_0.1_diff_frames'
	train_model_dir = 'model_checkpoints/' + model_run_name
	if not os.path.exists(train_model_dir):
		os.makedirs(train_model_dir)
	batch_size = 128
	resize_height, resize_width = 144, 256

	learning_rate = 1e-6
	regularization_strength = 0.1
	tf.reset_default_graph()
	#is_training = tf.placeholder(tf.bool, name='is_training')
	with tf.device(device):
		# Construct the computational graph we will use to train the model. We
		# use the model_init_fn to construct the model, declare placeholders for
		# the data and labels
		x = tf.placeholder(tf.float32, [None, resize_height, resize_width, 3])
		y = tf.placeholder(tf.int32, [None])
		
		# We need a place holder to explicitly specify if the model is in the training
		# phase or not. This is because a number of layers behaves differently in
		# training and in testing, e.g., dropout and batch normalization.
		# We pass this variable to the computation graph through feed_dict as shown below.
		#params = init_fn()
		# Use the model function to build the forward pass.
		is_training = tf.placeholder(tf.bool, name='is_training')
		scores = model_init_fn(x)
		# Compute the loss like we did in Part II

		#loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
		#loss = tf.reduce_mean(loss)
		# scores = model_init_fn(x)

		# Compute the loss like we did in Part II
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
		loss = tf.reduce_mean(loss)

		#regularization
		reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		loss += regularization_strength * sum(reg_losses)
		# Use the optimizer_fn to construct an Optimizer, then use the optimizer
		# to set up the training step. Asking TensorFlow to evaluate the
		# train_op returned by optimizer.minimize(loss) will cause us to make a
		# single update step using the current minibatch of data.
		
		# Note that we use tf.control_dependencies to force the model to run
		# the tf.GraphKeys.UPDATE_OPS at each training step. tf.GraphKeys.UPDATE_OPS
		# holds the operators that update the states of the network.
		# For example, the tf.layers.batch_normalization function adds the running mean
		# and variance update operators to tf.GraphKeys.UPDATE_OPS.
		optimizer = optimizer_init_fn()
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss)

	total_parameters = 0
	for variable in tf.trainable_variables():
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		#print(shape)
		#print(len(shape))
		variable_parameters = 1
		for dim in shape:
		#	print(dim)
			variable_parameters *= dim.value
		#print(variable_parameters)
		total_parameters += variable_parameters
	print("Total parameters: ", total_parameters)


	saver = tf.train.Saver()

	#print_ever = 100

	iterations_per_epoch = 100

	with tf.Session() as sess:

		# if not os.path.exists(train_model_dir):
  #       	os.makedirs(train_model_dir)
		sess.run(tf.global_variables_initializer())
		# else:
		# 	saver.restore(sess, "/tmp/model.ckpt")

		for epoch in range(num_epochs):
			print("Starting epoch: ", epoch)
			for i in range(iterations_per_epoch):
				print("Iteration ", epoch * iterations_per_epoch + i)
				curr_time = time.time()

				#x_np, y_np = load_single_frame_batch(batch_size)
				x_np, y_np = load_batch_multiple_frames_into_single(batch_size)


				feed_dict = {x: x_np, y: y_np, is_training:1}
				loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)	

				new_time = time.time()
				print("Iteration took: ", new_time - curr_time, " seconds.")
				print("Loss: ", loss_np)
			curr_time = time.time()
			check_accuracy_single_frame(sess, x, scores, is_training = is_training)
			new_time = time.time()
			print("Validation Check took: ", new_time - curr_time)
			curr_time = time.time()
			check_accuracy_single_frame(sess, x, scores, dataset = 'train', is_training = is_training)
			new_time = time.time()
			print("Training Check took: ", new_time - curr_time)
		#if epoch % 200 == 0:
			save_path = saver.save(sess, train_model_dir + "training_iteration_" + str((epoch + 1) * 100))



	return


def optimizer_init_fn():
	learning_rate = 1e-6
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	return optimizer


# Function to train given compiled model on training data
# Inputs:
#	model: compiled model from function such as vgg_model_init
#	x_train, y_train, x_val (optional), y_val (optional)
#	batch_size (optional): if None, model.fit defaults to 32
#	num_epochs: number of epochs
# Returns:
#	model: trained model, ready to predict with
#	history: history object from model.fit method
def train(model, x_train, y_train, num_epochs, x_val=None, y_val=None, batch_size=None):

	# Define validation data
	val_data = None

	#if x_val != None and y_val != None:
#		val_data = (x_val, y_val)
	print("shape of x train: ", x_train.shape)
	print("shape of y_train: ", y_train.shape)
	print("shape of x val: ", x_val.shape)
	print("shape of y_val: ", y_val.shape)
	# Use model.fit method to train model
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs) #, validation_data=val_data) # NOTE: not entirely sure validation_data param can be set to None

	return model, history



# def better_train_sess(model_init_fn, optimizer, num_epochs, image_set_size):



def check_accuracy(sess, x, scores, dataset = 'validation', is_training=None, check_big = False):
	"""
	Check accuracy on a classification model.
	
	Inputs:
	- sess: A TensorFlow Session that will be used to run the graph
	- dset: A Dataset object on which to check accuracy
	- x: A TensorFlow placeholder Tensor where input images should be fed
	- scores: A TensorFlow Tensor representing the scores output from the
	  model; this is the Tensor we will ask TensorFlow to evaluate.
	  
	Returns: Nothing, but prints the accuracy of the model


	"""

	batch_size = 2
	image_set_size = 12
	skip_frames = 8
	number_batches_check = 25
	if check_big:
		number_batches_to_check = 2500 // 2
	num_correct, num_samples = 0, 0
	pred_cumulative = ''
	actual_cumulative = ''
	for i in range(number_batches_check):
		x_batch, y_batch = load_batch(batch_size, image_set_size, skip_frames, dataset = dataset)
		feed_dict = {x: x_batch, is_training: 0}
		scores_np = sess.run(scores, feed_dict=feed_dict)
		y_pred = scores_np.argmax(axis=1)
		num_samples += x_batch.shape[0]
		num_correct += (y_pred == y_batch).sum()
		# with open("060318_bigger_image_results.txt", "a") as myfile:	
		# 	myfile.write("predicted: " + str(y_pred))
		# 	myfile.write("\n")
		# 	myfile.write("actual: " + str(y_batch))
		# 	myfile.write("\n")
		pred_cumulative += str(y_pred)
		actual_cumulative += str(y_batch)


	with open("060318_bigger_image_results_data_augmented.txt", "a") as myfile:
		if dataset == 'validation':
			myfile.write("Checking validation set.")
		else:
			myfile.write("Checking training set.")	
		myfile.write("\n")	
		if check_big:
			myfile.write("Performing check over a large portion of validation set")	
			myfile.write("\n")	
		myfile.write("Predicted: " + pred_cumulative)
		myfile.write("\n")
		myfile.write("Actual: " + actual_cumulative)
		myfile.write("\n")


	acc = float(num_correct) / num_samples
	print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

	with open("060318_bigger_image_results_data_augmented.txt", "a") as myfile:
		myfile.write("Accuracy " + str(acc))
		myfile.write("\n")
		#myfile.write("")
	#if check_big:
	#	return acc

def train_part34(model_init_fn, optimizer_init_fn, num_epochs=10):
	"""
	Simple training loop for use with models defined using tf.keras. It trains
	a model for one epoch on the CIFAR-10 training set and periodically checks
	accuracy on the CIFAR-10 validation set.
	
	Inputs:
	- model_init_fn: A function that takes no parameters; when called it
	  constructs the model we want to train: model = model_init_fn()
	- optimizer_init_fn: A function which takes no parameters; when called it
	  constructs the Optimizer object we will use to optimize the model:
	  optimizer = optimizer_init_fn()
	- num_epochs: The number of epochs to train for
	
	Returns: Nothing, but prints progress during trainingn
	"""

	batch_size = 2
	image_set_size = 12
	skip_frames = 8
	#iresize_height, resize_width = 216, 
	tf.reset_default_graph()    
	with tf.device(device):
		# Construct the computational graph we will use to train the model. We
		# use the model_init_fn to construct the model, declare placeholders for
		# the data and labels
		x = tf.placeholder(tf.float32, [None, image_set_size, resize_height, resize_width, 3])
		y = tf.placeholder(tf.int32, [None])
		
		# We need a place holder to explicitly specify if the model is in the training
		# phase or not. This is because a number of layers behaves differently in
		# training and in testing, e.g., dropout and batch normalization.
		# We pass this variable to the computation graph through feed_dict as shown below.
		is_training = tf.placeholder(tf.bool, name='is_training')
		
		# Use the model function to build the forward pass.
		scores = model_init_fn(x)

		# Compute the loss like we did in Part II
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
		loss = tf.reduce_mean(loss)

		# Use the optimizer_fn to construct an Optimizer, then use the optimizer
		# to set up the training step. Asking TensorFlow to evaluate the
		# train_op returned by optimizer.minimize(loss) will cause us to make a
		# single update step using the current minibatch of data.
		
		# Note that we use tf.control_dependencies to force the model to run
		# the tf.GraphKeys.UPDATE_OPS at each training step. tf.GraphKeys.UPDATE_OPS
		# holds the operators that update the states of the network.
		# For example, the tf.layers.batch_normalization function adds the running mean
		# and variance update operators to tf.GraphKeys.UPDATE_OPS.
		optimizer = optimizer_init_fn()
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss)

	# Now we can run the computational graph many times to train the model.
	# When we call sess.run we ask it to evaluate train_op, which causes the
	# model to update.
	#x_train, y_train, x_val, y_val, x_test, y_test = load_data(None)

	#train_data, val_data = load_data_new(None)



	total_parameters = 0
	for variable in tf.trainable_variables():
		# shape is an array of tf.Dimension
		shape = variable.get_shape()
		#print(shape)
		#print(len(shape))
		variable_parameters = 1
		for dim in shape:
		#	print(dim)
			variable_parameters *= dim.value
		#print(variable_parameters)
		total_parameters += variable_parameters
	print("Total parameters: ", total_parameters)


	saver = tf.train.Saver()

	print_every = 50

	num_epochs = 5000

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		t = 1
		for epoch in range(num_epochs):
			print('Starting epoch %d' % epoch)
			curr_time = time.time()
			# for x_np, y_np in train_dset:
			#     feed_dict = {x: x_np, y: y_np, is_training:1}
			#     loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
			#     if t % print_every == 0:
			#         print('Iteration %d, loss = %.4f' % (t, loss_np))
			#         check_accuracy(sess, val_dset, x, scores, is_training=is_training)
			#         print()
			#     t += 1
			x_np, y_np = load_batch(batch_size, image_set_size, skip_frames)
			print("loaded batch")
			feed_dict = {x: x_np, y: y_np, is_training:1}
			loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)
			#print("sess is running?")

			new_time = time.time()

			print("Epoch took: ", new_time - curr_time, " seconds.")
			print("Loss: ", loss_np)
			if t % print_every == 0:
				#print('Iteration %d, loss = %.4f' % (t, loss_np))
				curr_time = time.time()
				check_accuracy(sess, x, scores, is_training=is_training)
				new_time = time.time()
				print("Validation Check took: ", new_time - curr_time, " seconds.")
				#print()
				curr_time = time.time()
				check_accuracy(sess, x, scores, is_training=is_training, dataset = 'train')
				new_time = time.time()
				print("Training Check took: ", new_time - curr_time, " seconds.")
			t += 1
			if t % 2000 == 0:
				print("Performing large validation check and saving to file...")
				curr_time = time.time()
				check_accuracy(sess, x, scores, is_training=is_training, check_big = True)
				new_time = time.time()
				print("Big Validation Check took: ", new_time - curr_time, " seconds.")

			#print("end of one thing")
			if epoch % 200 == 0:
				save_path = saver.save(sess, "model_checkpoints/conv3d_bigger_image_060318_data_augmented_better_printing" + str(epoch))


def check_accuracy_entire_dataset(sess, x, scores, dataset, is_training = None):

	image_names = read_text_file(dataset + '.txt')

	all_y_pred = []
	all_y_actual = []
	num_correct, num_total = 0, 0
	shuffle(image_names)
	batch_size = 100

	image_names = images_names[:200]
	for i in range(len(image_names) // 100 - 1):
		image_file_names = image_names[i * 100 : (i + 1) * 100]
		x_batch, y_batch = load_single_frame_batch(batch_size, image_names = image_file_names, dataset = dataset)
		feed_dict = {x: x_batch, is_training: 0}
		scores_np = sess.run(scores, feed_dict=feed_dict)
		y_pred = scores_np.argmax(axis=1)
		num_total += x_batch.shape[0]
		num_correct += (y_pred == y_batch).sum()

		all_y_pred += y_pred.tolist()
		all_y_actual += y_batch.tolist()
	F1_score = f1_score(all_y_actual, all_y_pred, average = 'micro')


	acc = num_correct / num_total
	print("F1 Score: ", F1_score)
	print('Got %d / %d correct (%.2f%%)' % (num_correct, num_total, 100 * acc))
	return 


def official_evaluation(model_init_fn, model_location, dataset, is_training = None):
	tf.reset_default_graph()
	resize_height, resize_width = 144, 256
	batch_size = 128
	with tf.device(device):
		# Construct the computational graph we will use to train the model. We
		# use the model_init_fn to construct the model, declare placeholders for
		# the data and labels
		x = tf.placeholder(tf.float32, [None, resize_height, resize_width, 3])
		y = tf.placeholder(tf.int32, [None])
		scores = model_init_fn(x)
		is_training = tf.placeholder(tf.bool, name='is_training')
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
		loss = tf.reduce_mean(loss)

		# Use the optimizer_fn to construct an Optimizer, then use the optimizer
		# to set up the training step. Asking TensorFlow to evaluate the
		# train_op returned by optimizer.minimize(loss) will cause us to make a
		# single update step using the current minibatch of data.
		
		# Note that we use tf.control_dependencies to force the model to run
		# the tf.GraphKeys.UPDATE_OPS at each training step. tf.GraphKeys.UPDATE_OPS
		# holds the operators that update the states of the network.
		# For example, the tf.layers.batch_normalization function adds the running mean
		# and variance update operators to tf.GraphKeys.UPDATE_OPS.
		optimizer = optimizer_init_fn()
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss)

	#saver = tf.train.Saver()
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(model_location + '.meta')
		saver.restore(sess,tf.train.latest_checkpoint(model_location))
		#tf.initialize_all_variables().run()
		graph = tf.get_default_graph()
		x = graph.get_tensor_by_name("x:0")
		y = graph.get_tensor_by_name("y:0")
		loss = graph.get_tensor_by_name("loss:0")
		train_op = graph.get_tensor_by_name("train_op:0")

		x_np, y_np = load_single_frame_batch(batch_size)
		feed_dict = {x: x_np, y: y_np, is_training:1}
		loss_np, _ = sess.run([loss, train_op], feed_dict=feed_dict)			
		print("Model restored")
		check_accuracy_entire_dataset(sess, x, scores, dataset, is_training)


# def load_model_checkpoint(model_init_fn, model_location, dataset, is_training = None):

# 	model_location = 'model_checkpoints/first_model_test/first_model_16'

# 	tf.reset_default_graph()
# 	# saver = tf.train.Saver()

# 	# with tf.Session() as sess:
# 	# 	saver.restore(sess, model_location)
# 	# 	print("Model restored")

# 	sess=tf.Session()
# 	saver = tf.train.import_meta_graph(model_location + '.meta')
# 	saver.restore(sess,tf.train.latest_checkpoint(model_location))


# Function to use given model to predict on given data and return metrics (accuracy for now) on predictions
# Inputs:
#	model: trained model from function such as train
#	x, y (optional)
# Returns:
#	predictions: predicted classes for input x
#	accuracy: accuracy of predicted classes given true classes y
def predict(model, x, y=None):

	# Use model to make predictions
	predictions = model.predict(x)

	# Get accuracy
	accuracy = None
	if y != None:
		accuracy = np.mean(predictions == y)

	return predictions, accuracy
