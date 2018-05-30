import tensorflow as tf
import numpy as np
from load_data import *
from data_batch import Data

import time




device = '/cpu:0'
#device = '/gpu:0'



image_set_size = 8
FR = image_set_size
resize_height, resize_width = 144, 256
W = resize_height
H = resize_width
D = 3
num_classes = 10





