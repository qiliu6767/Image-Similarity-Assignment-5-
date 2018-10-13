import torch.utils.data as data

from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
from hingeLoss import TripletLoss
from model import resnet18
import torch.nn as nn


def default_loader(path):
	return Image.open(path).convert('RGB')

def default_flist_reader(flist):
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			query, pos, neg = line.strip().split(',')
			imlist.append((query, pos, neg))
	return imlist

class ImageFilelist(data.Dataset):
	def __init__(self,flist, 
				 query_transform = None, 
				 pos_transform = None,
				 neg_transform = None,
				 flist_reader = default_flist_reader, 
				 loader = default_loader):
		"""
		Args:
			root: where the images are stored
			flist: where the text file containing the names of images is stored
		"""
		self.imlist = flist_reader(flist)		
		self.query_transform = query_transform
		self.pos_transform = pos_transform
		self.neg_transform = neg_transform
		self.loader = loader

	def __getitem__(self, index):
		query, pos, neg = self.imlist[index]
		query = self.loader(os.path.join(query))
		pos = self.loader(os.path.join(pos))
		neg = self.loader(os.path.join(neg))
		if self.query_transform is not None:
			query = self.query_transform(query)
		if self.pos_transform is not None:
			pos = self.pos_transform(pos)
		if self.neg_transform is not None:
			neg = self.neg_transform(neg)
		
		return query, pos, neg

	def __len__(self):
		return len(self.imlist)
