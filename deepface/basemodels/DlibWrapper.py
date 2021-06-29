from deepface.basemodels.DlibResNet import DlibResNet

def loadModel(weight_dir):
	return DlibResNet(weight_dir)