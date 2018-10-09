import glob
import json
import random
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

def get_negative_images(all_images, image_names, num_neg_images):
	"""
	Args:
		all_images: all images in the training dataset
		images_names: images in a particular class
		num_neg_images: targeted number of negative images
	"""
	# Generate random numbers
	random_numbers = np.arange(len(all_images))
	np.random.shuffle(random_numbers)

	# Stipulate the range of num_neg_images
	if int(num_neg_images) > len(all_images) - 1:
		num_neg_images = len(all_images) - 1

	# Generate negative images list
	neg_count = 0
	negative_images = []
	for random_number in list(random_numbers):
		# Images that are not in the same class with the query image
		# are added to the negative images
		if all_images[random_number] not in image_names:
			neg_count += 1
			negative_images.append(all_images[random_number])
			if neg_count > int(num_neg_images) - 1:
				break
	return negative_images

def get_positive_images(image_name, image_names, num_pos_images):
	"""
	Args:
		image_name: name of a specific query image
		image_names: list of image names for a specific class
		num_pos_images: targeted number of positive images
	"""
	# Generate random numbers
	# Note the positive images are generated from the same class
	random_numbers = np.arange(len(image_names))
	np.random.shuffle(random_numbers)

	# Stipulate the range of num_pos_images
	if int(num_pos_images) > len(image_names) - 1:
		num_pos_images = len(image_names) - 1

	# Generate positive images
	pos_count = 0
	positive_images = []
	for random_number in list(random_numbers):
		# Images that are not identical to 
		# but in the same class with the query image are added to positive images
		if image_name != image_names[random_number]:
			positive_images.append(image_names[random_number])
			pos_count += 1
			if int(pos_count) > int(num_pos_images) - 1:
				break
	return positive_images

def triplet_sampler(dir_path, output_path, num_neg_images, num_pos_images):
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
			positive_images = get_positive_images(image_name, image_names, num_pos_images)
			
			# For each positive image, generate negative images
			for positive_image in positive_images:
				negative_images =  get_negative_images(all_images, set(image_names), num_neg_images)

				# For each negative image, generate triplet
				for negative_image in negative_images:
					triplets.append(query_image + ',')
					triplets.append(positive_image + ',')
					triplets.append(negative_image + '\n')
		
	# Write the generated triplet names into a new file
	f = open(os.path.join(output_path, "triplets.txt"), 'w')
	f.write(''.join(triplets))
	f.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Optional app description')
	parser.add_argument("--input_directory",
						help = "An argument for input directory")
	parser.add_argument("--output_directory",
						help = "An argument for output directory")
	parser.add_argument("--num_pos_images",
						help = "An argument for the number of positive images per query image")
	parser.add_argument("--num_neg_images",
						help = "An argument for the number of negative images per query image")
	args = parser.parse_args()

	if args.input_directory == None:
		print("Input directory path is required!")
		quit()
	elif args.output_directory == None:
		print("Output directory path is required!")
		quit()
	elif args.num_pos_images == None:
		print("Number of positive images is required!")
		quit()
	elif args.num_neg_images == None:
		print("Number of negative images is required!")
		quit()
	elif int(args.num_pos_images) < 1:
		print("Number of positive images cannot be less than 1!")
	elif int(args.num_neg_images) < 1:
		print("Number of negative images cannot be less than 1!")

	if not os.path.exists(args.input_directory):
		print(args.input_directory + " path does not exist!")
		quit()

	if not os.path.exists(args.output_directory):
		print(args.output_directory + " path does not exist!")
		quit()

	print("Input Directory: " + args.input_directory)
	print("Output Directory: " + args.output_directory)
	print("Number of positive images per query image: " + args.num_pos_images)
	print("Number of negative images per query image: " + args.num_neg_images)

	triplet_sampler(dir_path = args.input_directory,
					output_path = args.output_directory,
					num_neg_images = args.num_neg_images,
					num_pos_images = args.num_pos_images)

# train_dataset = datasets.ImageFolder(train_dir, 
# 									 transform = transforms.ToTensor())
# trainloader = data.DataLoader(train_dataset, 
# 							  batch_size = 1,
# 							  num_workers = 4)

# Test the triplet_sampler() function
# train_dir = "/Users/qiliu/Desktop/Fall2018/IE534/Homeworks/Homework05/tiny-imagenet-200/train"
# train_dir = "/projects/training/bauh/tiny-imagenet-200/train"
# output_path = "/Users/qiliu/Desktop/Fall2018/IE534/Homeworks/Homework05"
# output_path = '/u/training/tra415/HW05'
# triplet_sampler(dir_path = train_dir,
# 				output_path = output_path,
# 				num_neg_images = 1,
# 				num_pos_images = 1)
