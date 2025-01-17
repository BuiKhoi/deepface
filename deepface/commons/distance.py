import numpy as np

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findThreshold(model_name, distance_metric):

	base_threshold = {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75}

	thresholds = {
		'vgg_face': {'cosine': 0.40, 'euclidean': 0.55, 'euclidean_l2': 0.75},
		'open_face': {'cosine': 0.10, 'euclidean': 0.55, 'euclidean_l2': 0.55},
		'facenet':  {'cosine': 0.40, 'euclidean': 10, 'euclidean_l2': 0.80},
		'deep_face': {'cosine': 0.23, 'euclidean': 64, 'euclidean_l2': 0.64}
		}

	threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)

	return threshold
