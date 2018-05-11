import scipy
import numpy as np 


#global variables
dance_data_directory_name = 'tiny_dance_data'
directory_path = '../' + dance_data_directory_name + '/'
classes = ['ballet', 'break', 'flamenco', 'foxtrot', 'latin', 'quickstep', 'square', 'swing', 'tango', 'waltz']




#read in the txt file that contains path to each image

def read_text_file(file_name): #returns a list of paths to the images
	#file name is the name of text file with all the paths
	list_paths = []
	file_path = directory_path + file_name

	with open(file_path, 'r') as f:
		for line in f.readlines():
			list_paths.append(directory_path + line[:-1])

	return list_paths


from scipy.ndimage import imread

def read_image_file(file_path):
	#takes in the path to a image file and returns that image file as a 3-D numpy array (H, W, RGB)

	image_array = imread(file_path, flatten = False)
	return image_array


def find_label(file_path): 
	"""
	returns an int form of the label, ballet = 0, break = 1, etc.
	"""
	for klass_index in range(len(classes)):
		klass = classes[klass_index]
		if klass in file_path:
			return klass_index
	print("No Class Found, please look at file path")
	return

def read_all_images(dataset = "train"): #dataset should be train, validation, or test
	#takes in the dataset name and returns a dictionary of image path name to image as a np array
	#text_file_path = directory_path + dataset + '.txt'
	image_paths = read_text_file(dataset + '.txt')

	path_image_dict = {}
	i = 0
	for image_path in image_paths:
		print(i)
		image_array = read_image_file(image_path)
		path_image_dict[image_path] = image_array
		i += 1
	return path_image_dict




