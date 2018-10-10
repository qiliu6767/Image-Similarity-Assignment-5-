import torch.utils.data as data

from PIL import Image
import os
import os.path
import torchvision.transforms as transforms

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

train_transform = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.RandomCrop(32, padding = 4),
	# Convert the images into a format usable by PyTorch
	transforms.ToTensor(), 
	# Make all pixels range between -1 to +1
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainloader = data.DataLoader(
	ImageFilelist(flist = '/Users/qiliu/Desktop/Fall2018/IE534/Homeworks/Homework05/triplets.txt',
			      query_transform = train_transform,
				  pos_transform = train_transform,
				  neg_transform = train_transform),
	batch_size = 10,
	shuffle = False,
	num_workers = 4,
	pin_memory = True)

# # Test 
# print(len(list(trainloader)))
# print(type(trainloader))
# print(list(trainloader)[9999][2].size())






















