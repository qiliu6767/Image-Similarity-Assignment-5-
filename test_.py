import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np

def test(model, train_loader, val_loader, 
		 cuda_avail, val_image_labels, 
		 train_image_labels):
	'''
	Args:
		model: the trained model
		train_loader: dataloader object for training dataset with batch size 1
		val_loader: dataloader object for validation dataset with batch size 1
		cuda_avail: check whether GPU is available
		val_image_labels: list with labels for each validation image
		train_image_labels: list with labels for each training image 
	'''
	model.eval()
	acc = 0.0

	# Step 1: Calculate the image embeddings of all training images and store them as a numpy array
	for i, (image, label) in enumerate(train_loader):
		if cuda_avail:
			image = Variable(image.cuda())
		image_emb = model(image)
		image_emb = image_emb.detach().numpy()
		if i == 0:
			train_images_emb = image_emb
		else:
			train_images_emb = np.vstack(train_images_emb, image_emb)

	# Step 2: Calculate the image embedding of a specific validation image and store it as a numpy array
	for i, image in enumerate(val_loader):
		# Change image into a Variable object and send it to GPU
		if cuda_avail:
			image = Variable(image.cuda())

		# Calculate the image embedding
		image_emb = model(image)

		# Convert the validation embedding into a numpy
		image_emb = image_emb.detach().numpy()

		# Convert the validation embedding numpy to into the same shape with training image embeddings
		image_emb = np.tile(image_emb, (num_images, 1))

		# Calculate the difference
		diff_emb = image_emb - train_images_emb

		# Calculate L2 Norm for each row
		l2norm_emb = np.linalg.norm(diff_emb, ord = 2, axis = 1)

		# Get the smallest 100 l2norm
		top_similar = l2norm_emb.argsort()[:100]

		# Check the how many of these top similar images 
		# are in the same class with this current validation image
		val_label = val_image_labels[i]
		acc += sum(train_image_labels[top_similar] == val_label) / 100

	total_test_acc = acc / 10000
	return total_test_acc



	
