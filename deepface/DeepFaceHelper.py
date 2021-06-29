import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
from os import path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import dlib
import cv2

from deepface.detectors import DlibWrapper
from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID, ArcFace, Boosting
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons import functions, realtime, distance as dst

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8096)])

def build_model(model_name, weight_dir):

	"""
	This function builds a deepface model
	Parameters:
		model_name (string): face recognition or facial attribute model
			VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
			Age, Gender, Emotion, Race for facial attributes

	Returns:
		built deepface model
	"""

	global model_obj #singleton design pattern

	models = {
		'vgg_face': VGGFace.loadModel,
		'open_face': OpenFace.loadModel,
		'facenet': Facenet.loadModel,
		'deep_face': FbDeepFace.loadModel,
		'dlib': DlibWrapper.build_model,
		'age': Age.loadModel,
		'gender': Gender.loadModel
	}

	if not "model_obj" in globals():
		model_obj = {}

	if not model_name in model_obj.keys():
		model = models.get(model_name)
		if model:
			model = model(weight_dir)
			model_obj[model_name] = model
			print(model_name, "built")
		else:
			raise ValueError('Invalid model_name passed - {}'.format(model_name))

	return model_obj[model_name]

class DeepFaceHelper:
	def __init__(self, weights_dir=None, detector="dlib"):
		if weights_dir is not None:
            # Load embedding models
			self.embedding_models = ["vgg_face", "facenet", "open_face", "deep_face"]
			self.models = {}
			for em in self.embedding_models:
				self.models[em] = build_model(em, weights_dir)

			# Load detector:
			if detector != "dlib":
				raise NotImplementedError(f"Detector {detector} is not implemented")
			else:
				self.detector = build_model(detector, weights_dir)

			# Load age and gender model
			# self.models["age"] = build_model("age", weights_dir)
			# self.models["gender"] = build_model("gender", weights_dir)

			# Load light gbm model
			self.boosted_tree = Boosting.build_gbm(weights_dir)
		else:
			print("Weight directory is not specified, skipping")

	def get_embedding(self, image, model):
		input_shape_x, input_shape_y= functions.find_input_shape(model)
		preprocessed_image = functions.preprocess_face(image, target_size=(input_shape_y, input_shape_x))

		return model.predict(preprocessed_image)[0].tolist()
    
	def get_face_embedding(self, image_path):
		image = functions.load_image(image_path)

		# Detect and align face
		aligned_face = functions.detect_and_align(image, self.detector, "dlib", grayscale=False, enforce_detection=True, align=True)

		# Predict embeddings
		embeddings = {}
		for em in self.embedding_models:
			# print("Predicting", em)
			embeddings[em] = self.get_embedding(aligned_face, self.models[em])

		return embeddings

	def get_distances(self, embd1, embd2):
		distances = {}
		for em in self.embedding_models:
			distances[em] = {
				"cosine": dst.findCosineDistance(embd1[em], embd2[em]),
				"euclidean": dst.findEuclideanDistance(embd1[em], embd2[em]),
				"euclidean_l2": dst.findEuclideanDistance(
					dst.l2_normalize(embd1[em]),
					dst.l2_normalize(embd2[em])
				)
			}
		return distances

	def is_valid_face_thresholding(self, distances):
		valid = {}
		approveds = []
		for model_name in distances:
			temp_valid = {}
			for dist_name in distances[model_name]:
				threshold = dst.findThreshold(model_name, dist_name)
				approved = distances[model_name][dist_name] <= threshold
				temp_valid[dist_name] = approved
				approveds.append(approved)
			valid[model_name] = temp_valid
		return valid, sum(approveds)/len(approveds)

	def is_valid_face_ensemble(self, distances):
		ensemble_features = [
			distances["vgg_face"]["cosine"],
			distances["vgg_face"]["euclidean"],
			distances["vgg_face"]["euclidean_l2"],

			distances["facenet"]["cosine"],
			distances["facenet"]["euclidean"],
			distances["facenet"]["euclidean_l2"],

			distances["open_face"]["cosine"],
			distances["open_face"]["euclidean_l2"],

			distances["deep_face"]["cosine"],
			distances["deep_face"]["euclidean"],
			distances["deep_face"]["euclidean_l2"]
		]

		prediction = self.boosted_tree.predict(np.expand_dims(np.array(ensemble_features), axis=0))[0]

		verified = np.argmax(prediction) == 1
		score = prediction[np.argmax(prediction)]

		resp_obj = {
			"verified": verified,
			"score": score,
			"distance": ensemble_features,
			"model": self.embedding_models,
			"similarity_metric": distances["vgg_face"].keys()
		}

		return resp_obj
