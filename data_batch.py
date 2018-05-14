import numpy as np
import random

from load_data import *

"""

This file should contain the data class.
The data class will have all the data stored which we can then extract.

"""




class Data(object):

	def __init__(self, paths_to_images_dict, batch_type = 5):
		"""
		Inputs: 
		paths_to_images_dict is a dictionary of image path to image array stored as 3-D np array
		batch_type tells it what kind of batch we want:
			5 means 5-D tensor of Batch x Image Set x Height x Width x RGB
			4 means 4-D tensor of Batch x Image x Height x Width x RGB

		"""

		self.path_image_dict = paths_to_images_dict
		self.batch_type = batch_type

		#create dict of path name to label

		self.path_label_dict = {}

		#create a dict of group (video identifier) to label
		self.video_label_dict = {}

		for path in self.path_image_dict:
			label = find_label(path)
			self.path_label_dict[path] = label

			#NOTE: I think here looking at the first 6 characters is enough
			self.video_label_dict[path[:6]] = label

		#maximum number of frames a single video may have
		self.max_frame_count = 300



	def get_random_image(self):
		random_path = random.choice(list(self.path_image_dict))
		array = self.path_image_dict[random_path]
		label = self.path_label_dict[random_path]
		return array, label

	def get_random_image_set(self, image_set_size):

		random_video = random.choice(list(self.video_label_dict))
		label = self.video_label_dict[random_video]

		relevant_frames = [x for x in self.path_image_dict if random_video in x] #extract all the keys for the chosen video
		relevant_frames.sort()
		random_start = random.randint(0, self.max_frame_count - image_set_size) #do I need to subtract 1?

		relevant_frames = relevant_frames[random_start: random_start + image_set_size]

		frames = []
		for frame_path in relevant_frames:
			frames.append(self.path_image_dict[frame_path])

		array = np.stack(frames, axis = 0)
		#array should be a 4-D tensor
		return array, label

	def create_batch(self, batch_size = 4):
		if self.batch_type == 4:
			images = []
			labels = []
			for i in range(batch_size):
				image, label = get_random_image
				images.append(image)
				labels.append(label)
			batch = np.stack(images, axis = 0)
			batch_labels = np.asarray(labels)

		elif self.batch_type == 5:
			images = []
			labels = []
			for i in range(batch_size):

				image_set, label = self.get_random_image_set(image_set_size = 8)
				images.append(image_set)
				labels.append(label)

			batch = np.stack(images, axis = 0)
			batch_labels = np.asarray(labels)
			

		#batch should be 4 or 5-D tensor, batch_labels should be 1-D, batch_size X 1
		return batch, batch_labels	










