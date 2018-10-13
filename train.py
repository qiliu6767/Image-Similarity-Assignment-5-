# Import packages and functions
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os

# Import custom defined functions and modules
from model import resnet18
import ImageGenerator
import tripletsampler
from hingeLoss import TripletLoss
from test_ import test
from tripletsampler import list_images

#------------------------------------------------
# Parameter for generating triplets.txt file
train_dir = "tiny-imagenet-200/train"
output_path = ""

# Transformation for DataLoader object
train_transform = transforms.Compose([
	transforms.RandomHorizontalFlip(),
	transforms.RandomResizedCrop(224),
	# Convert the images into a format usable by PyTorch
	transforms.ToTensor(), 
	# Make all pixels range between -1 to +1
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform = transforms.Compose([
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#------------------------------------------------
# Load model and change the number of final outputs
model = resnet18(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4096)

# Function for check GPU support
cuda_avail = torch.cuda.is_available()
if cuda_avail:
	model.cuda()

# Function for saving and evaluating the model
def save_model(epoch):
	torch.save(model.state_dict(), "deepRanking_{}.model".format(epoch))
	print("Checkpoint saved")

#------------------------------------------------
# Optimizer and loss function
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
loss = TripletLoss()

#------------------------------------------------
# Function for updating learning rate
def adjust_learning_rate(epoch):
	lr = 0.001
	if epoch > 180:
		lr = lr / 1000000
	elif epoch > 150:
		lr = lr / 100000
	elif epoch > 120:
		lr =  lr / 10000
	elif epoch > 90:
		lr = lr / 1000
	elif epoch > 60:
		lr = lr / 100
	elif epoch > 30:
		lr = lr / 10

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

#------------------------------------------------
# Data preparation for test procedure
# Create a list for labels of validation images
val_image_labels = []
val_annotations_path = "tiny-imagenet-200/val/val_annotations.txt"
with open(val_annotations_path, 'r') as val_labels:
	for line in val_labels.readlines():
		image, label, _, _, _, _ = line.strip().split('\t')
		val_image_labels.append(label)

# Generate a list for labels of training images
classes = [d for d in os.listdir(train_dir)
			 if os.path.isdir(os.path.join(train_dir, d))]
train_image_labels = [label for label in classes for i in range(500)]
print(classes)

# Make a train_loader for the training images
train_dataset = datasets.ImageFolder(train_dir,
									 transform = train_transform)
train_loader = data.DataLoader(train_dataset,
							   batch_size = 1,
							   num_workers = 3)

# Make a val_loader for the validation images
val_dir = "tiny-imagenet-200/val"
val_dataset = datasets.ImageFolder(val_dir, 
								   transform = val_transform)
val_loader = data.DataLoader(val_dataset, 
							 batch_size = 1, # Because each image should be passed into the model respectively
							 num_workers = 3)
num_images = 100000

#------------------------------------------------
# Function for training model
def train(model, num_epochs):
	'''
	model: model being trained, i.e. the pretrained resnet18 model
	'''
	train_loss = []

	# Iteration for each epoch
	for epoch in range(num_epochs):
		# Step 1: Generate the new triplets.txt file for this epoch
		tripletsampler.triplet_sampler(train_dir, output_path)

		# Step 2: Generate Dataloader object w.r.t the new triplets.txt
		flist_triplets = 'triplets.txt'
		trainloader = data.DataLoader(
			ImageGenerator.ImageFilelist(flist = flist_triplets,
			      					 query_transform = train_transform,
				  					 pos_transform = train_transform,
				  					 neg_transform = train_transform),
			batch_size = 100,
			shuffle = False,
			num_workers = 4,
			pin_memory = True)

		N = len(list(trainloader))

		# Step 3: Calculate embeddings and loss
		# Set the model to training mode
		model.train()

		# Initialize training loss to sum up loss w.r.t each batch
		train_loss_batch = 0.0

		# Iteration for each batch
		for i, (q, p, n) in enumerate(trainloader):
			if cuda_avail:
				q = Variable(q.cuda())
				p = Variable(p.cuda())
				n = Variable(n.cuda())

			# Clear all the accumulated gradients
			optimizer.zero_grad()

			# Calculate image embeddings for q, p, n
			q_emb = model(q)
			p_emb = model(p)
			n_emb = model(n)

			# # Update the train_images_emb
			# if i == 0:
			# 	train_images_emb = q_emb
			# else:
			# 	train_images_emb = np.vstack(train_images_emb, q_emb)

			# Code for preventing overflow error
			if (epoch > 2):
				for group in optimizer.param_groups:
					for p in group['params']:
						state = optimizer.state[p]
						if (state['step'] >= 1024):
							state['step'] = 1000 

			# Calculate hinge loss
			loss = loss(q_emb, p_emb, n_emb)
			train_loss_batch += loss

			# Backpropagation
			loss.backward()

			# Update parameters
			optimizer.step()

			# Update learning rate
			adjust_learning_rate(epoch)

		# Calculate the average training loss over the entire training set
		train_loss_batch = train_loss_batch / N
		train_loss.append(train_loss_batch)

		# In each epoch, after updating the model, evaluate the model
		test_acc = test(model, train_loader, val_loader, 
		 				cuda_avail, val_image_labels, 
		 				train_image_labels)

		print('Epoch {}, Train Loss: {}'.format(epoch, train_loss_batch))
		print('Epoch {}, Test Accuracy: {}'.format(epoch, test_acc))

	return train_loss

# # Train the model
# if __name__ == '__main__':
# 	train(model = model, num_epochs = 40)