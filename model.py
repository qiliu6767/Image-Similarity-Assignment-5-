import torch
import torchvision
import torch.utils.model_zoo as model_zoo
import torch.nn as nn


# Function for loading the pretrained model
def resnet18(pretrained = True):
	model_url = {'resnet18': 'http://download.pytorch.org/models/resnet18-5c106cde.pth'}
	model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
	# Load weights
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_url['resnet18'], model_dir = './'))
	return model
