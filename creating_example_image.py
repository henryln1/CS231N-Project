import scipy
import numpy as np 
import random
from scipy.ndimage import imread


#image_file_name = '../dance_data/validation/waltz/AQ-izQC-S1Y_031_0049.jpg'


image_file_name = '_rHO5g9KPyI_020_0038.jpg'
image_array = imread(image_file_name, flatten = False)

resize_height = 216
resize_width = 384
print("shape of image_array: ", image_array.shape)
resized_image = scipy.misc.imresize(image_array, (resize_height, resize_width))


scipy.misc.imsave("reduced_image_break_1_216x384.jpg", resized_image)