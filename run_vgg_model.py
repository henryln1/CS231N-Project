# CS231N Final Project
# Henry Lin (henryln1@stanford.edu) and Christopher Vo (cvo9@stanford.edu)
# Program to load data, initialize, train, and predict on VGG-inspired model


# Imported needed libraries and classes
import vgg_model as vgg
import tensorflow as tf

from load_data import *

#device = '/gpu:0'

# Define various parameters
data_path = ''
num_classes = 10
batch_size = 64
num_epochs = 10
output_path = None


# Function to write predictions and accuracy to output files
# Inputs:
#	predictions: predictions on test set
#	accuracy: test accuracy
#	output_path: name of output directory, if None, defaults to current dir
# Returns:
#	pred_output, acc_output: output files of predictions and accuracy to specified directory
def createOutputFiles(predictions, accuracy, output_path):

	# Modify output path accordingly
	if output_path != None:
		output_path += '/'
	else:
		output_path = ''

	# Create output files
	pred_output_name = "test_predictions.txt"
	acc_output_name = "test_accuracy.txt"

	# Predictions output file
	pred_output = open(output_path + pred_output_name, "w")
	pred_output.write("Test Set Predictions\n")

	for pred in predictions:
		pred_output.write(str(pred) + "\n")

	pred_output.close()

	# Accuracy Output File
	acc_output = open(output_path + acc_output_name, "w")
	acc_output.write("Test Set Accuracy: " + str(accuracy) + "\n")
	acc_output.close()





# Main function
def main():

	# if not os.path.exists(FLAGS.train_dir):
 #            os.makedirs(FLAGS.train_dir)
 #        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
 #        logging.getLogger().addHandler(file_handler)

	vgg.train_part34(vgg.vgg_model_init, vgg.optimizer_init_fn, num_epochs = 10)
	



	# # Load data
	# x_train, y_train, x_val, y_val, x_test, y_test = vgg.load_data(data_path)
	# N, FR, W, H, D = x_train.shape
	# input_shape = (FR, W, H, D)

	# # Initialize VGG model
	# vgg_model = vgg.vgg_model_init(input_shape, num_classes)

	# # Train model
	# vgg_model, history = vgg.train(vgg_model, x_train, y_train, num_epochs=num_epochs, x_val=x_val, y_val=y_val, batch_size=batch_size)

	# # Predict on test set and record accuracy
	# test_predictions, test_acc = vgg.predict(vgg_model, x_test, y_test)

	# # Write predictions and accuracy to output files
	# createOutputFiles(predictions, accuracy, output_path)

	# # Save model to output file to load again
	# vgg_model.save('vgg_model.h5')


def main2():
	#vgg.train_part34_single_image(vgg.vgg_model_single_image_init, vgg.optimizer_init_fn, num_epochs = 100)
	vgg.train_part34_single_image(vgg.vgg_model_single_image_init_non_seq, vgg.optimizer_init_fn, num_epochs = 100)
	
	#model_location = 
	#run_official_evaluation(model_location, 'validation')

def main3():
	#model_location = 'model_checkpoints/first_model_test/first_model_20'
	model_location = 'model_checkpoints/single_frames_model_batch_128_reg_strength_0.1_nonseqtraining_iteration_4800'
	dataset = 'train'
	model_init = vgg.vgg_model_single_image_init_non_seq
	vgg.official_evaluation(model_init, model_location , dataset = dataset)


def main4():
	vgg.train_part34(vgg.vgg_model_single_image_init_non_seq_lstm, vgg.optimizer_init_fn, num_epochs = 100)

def main5():
	vgg.train_part34(vgg.vgg_model_conv3d_init, vgg.optimizer_init_fn, num_epochs = 50)
# Allows for script to be run as program
if __name__=="__main__":
	main5()
