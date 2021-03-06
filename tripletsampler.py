import glob
import json
import csv
import os
import re
import argparse
import numpy as np

import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable

import time

def list_images(directory, ext = ['.JPG', '.JPEG', '.BMP', '.PNG', '.PPM']):
	"""
	Return a list of image names in the specified directory
	"""
	return [os.path.join(root, f) 
			for root, _, files in os.walk(directory) for f in files
			if os.path.splitext(f)[1] in ext]

def get_negative_images(all_images, image_names):
	"""
	Args:
		all_images: all images in the training dataset
		images_names: images in a particular class
	"""
	# Generate negative images list
	negative_image = ''

	while True:
		# Generate a random number
		random_number = np.random.randint(low = 0, high = len(all_images))
		if all_images[random_number] not in image_names:
			negative_image += all_images[random_number]
			break
	return negative_image

def get_positive_images(image_name, image_names):
	"""
	Args:
		image_name: name of a specific query image
		image_names: list of image names for a specific class
	"""
	# Generate positive images
	positive_image = ''

	while True:
		# Generate random number
		random_number = np.random.randint(low = 0, high = len(image_names))
		if image_name != image_names[random_number]:
			positive_image += image_names[random_number]
			break
	return positive_image

def triplet_sampler(dir_path, output_path):
	"""
	Args:
		dir_path: directory path where the training dataset lies
		output_path: directory path where the generated triplet is stored
	"""
	# List of all the class names
	classes = [d for d in os.listdir(dir_path)
			   if os.path.isdir(os.path.join(dir_path, d))]

	# Create list of all the images
	all_images = []
	for class_ in classes:
		all_images += (list_images(directory = os.path.join(dir_path, class_)))

	# Create the triplets
	triplets = []
	for class_ in classes:
		# Image names in this class
		image_names = list_images(directory = os.path.join(dir_path, class_))
		
		# For each query image, generate the positive images
		for image_name in image_names:
			# image_names_set = set(image_names)
			query_image = image_name
			positive_image = get_positive_images(image_name, image_names)
			negative_image = get_negative_images(all_images, set(image_names))
			triplets.append(query_image + ',')
			triplets.append(positive_image + ',')
			triplets.append(negative_image + '\n')
		
	# Write the generated triplet names into a new file
	f = open(os.path.join(output_path, "triplets.txt"), 'w')
	f.write(''.join(triplets))
	f.close()

# # Test the function
# train_dir = 'tiny-imagenet-200/train'
# output_path = ''
# triplet_sampler(train_dir, output_path)