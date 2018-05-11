# CS231N Final Project
# Henry Lin (henryln1@stanford.edu) and Christopher Vo (cvo9@stanford.edu)
# Implementation of initial VGGNet-inspired model (scaled down)


# Import needed libraries and functions
import tensorflow as tf
import numpy as np
from load_data import *
from data_batch import Data


#device = '/gpu:0'

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
def vgg_model_init(input_shape, num_classes):

	# Define hyperparams, weight initializer, activation, regularization, loss function, and optimizer
	FR, W, H, D = input_shape

	num_filters = [16, 32, 64] # number of filters in each set of conv layers
	filter_size = 3 # 3x3 filters
	filter_stride = 1

	pool_size = 2 # 2x2 max pool
	pool_stride = 2

	initializer = tf.variance_scaling_initializer(scale=2.0) # initializer for weights
	activation = tf.nn.relu # ReLU for each Conv layer

	reg_strength = 0.001
	regularization = tf.contrib.layers.l2_regularizer(reg_strength) # L2 regularization for FC layer

	#loss = tf.nn..ftmax_cross_entropy_with_logits # NOTE: not sure if this loss function works with the Keras compile/fit model methods, may have to manually implement train method like in homework

	optimizer = tf.train.AdamOptimizer() # optimizer used is Adam


	# Define architecture as sequential layers
	layers = [

		# Conv Layer Set 1: 2 Conv layers (16 filters), 1 Pool layer
		tf.layers.Conv3D(input_shape=input_shape, filters=num_filters[0], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.Conv3D(filters=num_filters[0], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.MaxPooling3D(pool_size=pool_size, strides=pool_stride, padding='valid'),

		# Conv Layer Set 2: 2 Conv layers (32 filters), 1 Pool layer
		tf.layers.Conv3D(filters=num_filters[1], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.Conv3D(filters=num_filters[1], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.MaxPooling3D(pool_size=pool_size, strides=pool_stride, padding='valid'),

		# Conv Layer Set 3: 3 Conv layers (64 filters), 1 Pool layer
		tf.layers.Conv3D(filters=num_filters[2], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.Conv3D(filters=num_filters[2], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.Conv3D(filters=num_filters[2], kernel_size=[FR, filter_size, filter_size], strides=filter_stride, padding='same', activation=activation, kernel_initializer=initializer),
		tf.layers.MaxPooling3D(pool_size=pool_size, strides=pool_stride, padding='valid'),

		# FC Layer
		tf.layers.Flatten(),
		tf.layers.Dense(num_classes, kernel_initializer=initializer, kernel_regularizer=regularization)
	]


	# Initialize model and compile
	vgg_model = tf.keras.Sequential(layers)
	vgg_model.compile(loss='mean_squared_error', optimizer=optimizer)

	return vgg_model


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