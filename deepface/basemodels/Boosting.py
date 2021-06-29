# from deepface import DeepFace
from tqdm import tqdm
import os
from os import path
from pathlib import Path
import numpy as np
import gdown

def validate_model(model):
	#validate model dictionary because it might be passed from input as pre-trained
	found_models = []
	for key, value in model.items():
		found_models.append(key)
	
	if ('VGG-Face' in found_models) and ('Facenet' in found_models) and ('OpenFace' in found_models) and ('DeepFace' in found_models):
		#print("Ensemble learning will be applied for ", found_models," models")
		valid = True
	else:
		
		missing_ones = set(['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace']) - set(found_models)
		
		raise ValueError("You'd like to apply ensemble method and pass pre-built models but models must contain [VGG-Face, Facenet, OpenFace, DeepFace] but you passed "+str(found_models)+". So, you need to pass "+str(missing_ones)+" models as well.")

def build_gbm(weight_dir):
	
	#this is not a must dependency
	import lightgbm as lgb #lightgbm==2.3.1
	
	weight_path = weight_dir+"face-recognition-ensemble-model.txt"
	
	if os.path.isfile(weight_path) != True:
		print("face-recognition-ensemble-model.txt will be downloaded...")
		url = 'https://raw.githubusercontent.com/serengil/deepface/master/deepface/models/face-recognition-ensemble-model.txt'
		gdown.download(url, weight_path, quiet=False)
		
	ensemble_model_path = weight_path
	
	deepface_ensemble = lgb.Booster(model_file = ensemble_model_path)
	
	return deepface_ensemble
