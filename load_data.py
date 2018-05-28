import scipy
import numpy as np 
import random

#global variables
#dance_data_directory_name = 'tiny_dance_data'
dance_data_directory_name = 'dance_data'
directory_path = '../' + dance_data_directory_name + '/'
classes = ['ballet', 'break', 'flamenco', 'foxtrot', 'latin', 'quickstep', 'square', 'swing', 'tango', 'waltz']




#read in the txt file that contains path to each image

def read_text_file(file_name): #returns a list of paths to the images
	#file name is the name of text file with all the paths
	list_paths = []
	file_path = directory_path + file_name

	with open(file_path, 'r') as f:
		for line in f.readlines():
			if len(line) > 25:
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
	print("len of image paths: ", len(image_paths))
	print("image path example: ", image_paths[0])
	for image_path in image_paths:
		#if len(image_path) < 25:
		#	continue
		#print(i)
		image_array = read_image_file(image_path)
		path_image_dict[image_path] = image_array
		i += 1

		if i % 1000 == 0:
			print("current path: ", image_path)
			print("Image Number: ", i)
	return path_image_dict



def load_batch(batch_size, image_set_size, skip_frames, dataset = "train"):
	"""
	batch size: Number of randomly chosen image sets we want
	image set size: the number of frames we want per randomly chosen video
	skip_frames: how many frames we skip (do we want every 5 frames, 10 frames, etc.)
	dataset: which dataset are we drawing this batch from
	"""
	print("loading batch...")
	resize_height, resize_width = 144, 256

	image_paths = read_text_file(dataset + '.txt') #a list of all the possible image files

	# start_frames = []
	# for i in range(1, 51):
	# 	curr_string = '0' + str(i)
	# 	if len(curr_string) < 3:
	# 		curr_string = '0' + curr_string
	# 	curr_string += '.jpg'
	# 	#print(curr_string)
	# 	start_frames += [x for x in image_paths if curr_string in x]
	start_frames = [x for x in image_paths if '001.jpg' in x]
	#print("Number of start frames: ", len(start_frames))
	random_start_frames = random.sample(start_frames, batch_size)
	#print("Random Start Frames: ", random_start_frames)

	video_ids = [x[:-8] for x in random_start_frames]
	#print("Video IDs: ", video_ids)

	buffer_zeros = '000000'
 
	if image_set_size * skip_frames > 300:
		print("Warning: Your image set size and skip frame variables may exceed the 300 frames per video. Please double check your numbers")

	list_image_sets = [] #list that contains each image set, each image set is a image_set_size x H x W x 3 array
	Y_train = []
	for video in video_ids:
		#print("current video id: ", video)
		image_set = []
		random_start_between_0_100 = random.randint(0, 300 - 1 - image_set_size * skip_frames)
		#print("product", image_set_size * skip_frames)
		for i in range(1, (image_set_size) * skip_frames, skip_frames):
			#print("current image frame: ", i)
			buffer_zeros_curr = buffer_zeros + str(i + random_start_between_0_100)
			#print("buffer: ", )
			current_frame = buffer_zeros_curr[-4:]
			#print("current frame: ", current_frame)
			image_frame_id = video + current_frame + '.jpg'
			#print("Current Image Name: ", image_frame_id)
			image_file = read_image_file(image_frame_id)
			resized_image = scipy.misc.imresize(image_file, (resize_height, resize_width))
			image_set.append(resized_image)
			if i == 1:
				Y_train.append(find_label(image_frame_id))
		image_set_array = np.stack(image_set, axis = 0) #want to take a bunch of H x W x 3 arrays and stack them on top of each other
		#print("Shape of image set array: ", image_set_array.shape)
		list_image_sets.append(image_set)

	Y_train = np.asarray(Y_train)
	X_train = np.stack(list_image_sets, axis = 0)
	#print("shape of X_train: ", X_train.shape)
	#print("shape of Y_train: ", Y_train.shape)
	#print("batch loaded")
	return X_train, Y_train







