import os
import numpy as np
import pandas as pd
import cv2
import base64
from pathlib import Path

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	import keras
	from keras.preprocessing import image
elif tf_version == 2:
	from tensorflow import keras
	from tensorflow.keras.preprocessing import image

#--------------------------------------------------

def initialize_input(img1_path, img2_path = None):

	if type(img1_path) == list:
		bulkProcess = True
		img_list = img1_path.copy()
	else:
		bulkProcess = False

		if (
			(type(img2_path) == str and img2_path != None) #exact image path, base64 image
			or (isinstance(img2_path, np.ndarray) and img2_path.any()) #numpy array
		):
			img_list = [[img1_path, img2_path]]
		else: #analyze function passes just img1_path
			img_list = [img1_path]

	return img_list, bulkProcess

def loadBase64Img(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def load_image(img):

	exact_image = False
	if type(img).__module__ == np.__name__:
		exact_image = True

	base64_img = False
	if len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	#---------------------------

	if base64_img == True:
		img = loadBase64Img(img)

	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")

		img = cv2.imread(img)

	return img

def preprocess_face(img, target_size=(224, 224), grayscale = False):
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img = cv2.resize(img, target_size)
	#TODO: resize causes transformation on base image, you should add black pixels to rezie it to target_size

	img_pixels = image.img_to_array(img)
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	img_pixels /= 255 #normalize input in [0, 1]

	return img_pixels

def find_input_shape(model):

	#face recognition models have different size of inputs
	#my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.

	input_shape = model.layers[0].input_shape

	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]

	if type(input_shape) == list: #issue 197: some people got array here instead of tuple
		input_shape = tuple(input_shape)

	return input_shape
