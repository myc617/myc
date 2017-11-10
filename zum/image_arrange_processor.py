from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path
import sys
import csv
import re
import tarfile
import struct
import shutil as st
import numpy as np
import scipy.spatial.distance
import scipy.stats
from pandas import Series, DataFrame
import pandas as pd
import tensorflow as tf
from six.moves import urllib
from colorthief import ColorThief
from Tkinter import *
from PIL import ImageTk, Image
import time

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

BOTTLENECK_TENSOR_SIZE = 2048
MODL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_CHANNEL = 3

BOTTLENECK_TENSOR = 'pool_3/_reshape:0'
RESIZED_INPUT_TENSOR = 'ResizeBilinear:0'

FLAGS = None

def make_image_list(image_dir):
	"""
	make list of images which are placed at image_dir directory

	Args:
		image_dir: directory of target images

	Returns:
		image_list: names of images
	"""
	if not os.path.exists(image_dir):
		print("There is no " + image_dir + " directory")
		return None
	image_list = os.listdir(image_dir)
    
	return image_list

def get_distance(image1, image2, dist_type):
	"""
	calculate distance of two images
	"""
	try:
		if dist_type == 'euclidean':
			distance = scipy.spatial.distance.euclidean(image1, image2)
		elif dist_type == 'cosine':
			distance = scipy.spatial.distance.cosine(image1, image2)
	except ValueError:
		print('Use proper distance type')

	return distance

def visualize_result(id_list):
	"""
	visualize total result of image arrangement

	Args:
		id_list: ordered image list which will be placed sequentially

	Returns:
		tkinter canvas which shows image arrangement
	"""
	root = Tk()
	root.title('image arrangement result')
	out_frame = Frame(root)
	out_frame.pack()
	canvas = Canvas(out_frame, width = 1800, height = 10000, scrollregion=(0,0,1800,10000))
	ybar = Scrollbar(out_frame, orient=VERTICAL)
	ybar.pack(side=RIGHT, fill=Y)
	ybar.config(command=canvas.yview)
	xbar = Scrollbar(out_frame, orient=HORIZONTAL)
	xbar.pack(side=BOTTOM, fill=X)
	xbar.config(command=canvas.xview)
	canvas.config(width = 1800, height = 10000)
	canvas.config(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
	canvas.pack(side=LEFT, expand=True, fill=BOTH)
	image_list = []
	width_list = []
	width = 0
	height = 0
	extensions  = ['jpg', 'jpeg', 'JPG', 'JPEG']
	for i, image_id in enumerate(id_list):
		image_id = image_id.split('.')[0]
		image_id = os.path.join(FLAGS.image_dir, image_id)
		for extension in extensions:
			is_image = image_id + '.' + extension
			if os.path.exists(is_image):
				image_id = is_image
				break
		print(image_id.split('/')[-1] + '  ' + str(i) + 'th image')
		img = Image.open(image_id)
		[w,h] = img.size
		x = h / 170
		w = int(w/x)
		h = int(h/x)
		img = img.resize((w,h), Image.ANTIALIAS)
		img = ImageTk.PhotoImage(img)
		image_list.append(img)
		width_list.append(w)
		if (width + width_list[i-1] + w + 10) > 1800:
			width = 0
			height += 1
		elif i == 0:
			pass
		else:
			width += width_list[i-1] + 10
		canvas.create_image(width, 180*height, image=img, anchor=NW)
	root.mainloop()

def dist_ordered_list_of_id(dist_dataframe, target_id):
	"""
	make ordered list of images using distance values between images

	Args:
		dist_dataframe: distance value dataframe between each images
		target_id: image to put at front of all image (highest AVA score image)

	Returns:
		dist_id_list: ordered list of images using distance dataframe
	"""
	dist_id_list = []
	dist_id_list.append(target_id)
	while not dist_dataframe.empty:
		closest_id = dist_dataframe[target_id].idxmin()
		dist_id_list.append(closest_id)
		dist_dataframe = dist_dataframe.drop(target_id)
		dist_dataframe = dist_dataframe.drop(target_id, axis=1)
		target_id = closest_id

	return dist_id_list

def make_cluster_folder(cluster_dir, cluster):
	"""
	make cluster folder to visualize every elements of a cluster

	Args:
		cluster_dir: name of a cluster
		cluster: list of name of images that are included at same cluster
	"""
	os.makedirs(cluster_dir)
	extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
	for idx in cluster:
		idx = idx.split('.')[0]
		idx = os.path.join(FLAGS.image_dir, idx)
		for extension in extensions:
			is_image = idx + '.' + extension
			if os.path.exists(is_image):
				break
		if os.path.exists(is_image):
			st.copy(is_image, cluster_dir)
		else:
			pass

def image_filteration(length, image_dict):
	"""
	cut representative images at 400 number using ava score

	Args:
		length: hyperparameter to decide total number of representative images
		image_dict: name of all representative images

	Returns:
		list of representative images which are less than 400
	"""
	_400_or_less_images = []
	for i in range(length):
		best_score_image = min(image_dict, key=image_dict.get)
		_400_or_less_images.append(best_score_image)
		image_dict.pop(best_score_image)

	return _400_or_less_images

def make_distance_dataframe(image_list, feature_dict):
	"""
	make distance dataframe from a feature data

	Args:
		image_list: list of representative images
		feature_dict: feature used at making distance dataframe (v3 feature, color feature, ava score)

	Returns:
		distance_dataframe: dataframe of distance values
		distance_dict: dictionary of distance values
	"""
	distance_dict = {}
	# if feature value is ava score we do not calculate distance
	# we just put ava score value at dictionary
	if isinstance(feature_dict[image_list[0]], float):
		for image_id in image_list:
			distance_dict[image_id] = feature_dict[image_id]
	else:
		for image_id1 in image_list:
			distance_arr = []
			image1 = feature_dict[image_id1]
			for image_id2 in image_list:
				image2 = feature_dict[image_id2]
				if image_id1 != image_id2:
					distance = get_distance(image1, image2, 'cosine')
				else:
					distance = sys.float_info.max / 1000000 # 100000 for future calculation
				distance_arr.append(distance)
			distance_dict[image_id1] = distance_arr
	distance_dataframe = DataFrame(distance_dict, columns = image_list, index = image_list)

	return distance_dataframe, distance_dict

def main(_):

	v3_feature = {}
	color_feature = {}
	ava_score = {}
	image_list = make_image_list(FLAGS.feature_dir)

	# put every features of all images at different dictionary 
	start_time = time.time()
	for image in image_list:
		image_features = open(os.path.join(FLAGS.feature_dir, image))
		feature_dataframe = pd.read_csv(image_features, index_col=0)
		if feature_dataframe['ava_score'][0] < 0.2:
			v3_feature[image] = feature_dataframe['v3_feature']
			color_feature[image] = feature_dataframe['color_feature'][:3]
			ava_score[image] = feature_dataframe['ava_score'][0]
	end_time = time.time()
	print(len(v3_feature))
	print("feature import: " + str(end_time-start_time) + 'seconds')

	def input_fn():
		df = DataFrame(v3_feature.values())

		return tf.constant(df.as_matrix(), tf.float32, df.shape), None

	# make kMeans cluster object and run the clustering process
	start_time = time.time()
	kmeans = tf.contrib.learn.KMeansClustering(num_clusters=FLAGS.num_cluster, relative_tolerance=FLAGS.tolerance)
	_ = kmeans.fit(input_fn = input_fn)
	end_time = time.time()
	print("clustering: " + str(end_time-start_time) + 'seconds')

	# make a list of indexes to identify what cluster image belongs to
	idxs = list(kmeans.predict_cluster_idx(input_fn=input_fn))
	clusters = kmeans.clusters()
	keys = v3_feature.keys()
	high_score_images = {}

	start_time = time.time()
	for i in range(FLAGS.num_cluster):
		# make list of images that is included at same cluster
		cluster_dir = FLAGS.feature_dir.split('features')[0] + 'kmeans/Cluster-' + str(i)
		idxs_of_same_clust = [idx for idx, value in enumerate(idxs) if value == i]
		cluster = [keys[j] for j in idxs_of_same_clust]

		# find highest score image among the list of images
		# make a list of the highest score images
		scores_of_cluster = {}
		if cluster:
			for idx in cluster:
				#scores_of_cluster[idx] = ava_score[idx]
				scores_of_cluster[idx] = get_distance(clusters[i], v3_feature[idx], 'euclidean')
			best_score_image = min(scores_of_cluster, key=scores_of_cluster.get)
			high_score_images[best_score_image] = ava_score[best_score_image]
		else:
			continue

		# visualize cluster elements by making directory
		# you may skip this process
		if not os.path.exists(cluster_dir):
			make_cluster_folder(cluster_dir, cluster)
		else:
			tf.gfile.DeleteRecursively(cluster_dir)
			make_cluster_folder(cluster_dir, cluster)

	# if the number of cluster is less than 400, limit the representative images at number of the cluster
	# otherwise, limit the representative images at number of 400
	if len(high_score_images) > 400:
		_400_or_less_images = image_filteration(400, high_score_images)
	else:
		_400_or_less_images = image_filteration(len(high_score_images), high_score_images)

	# make distance dataframe for each feature of images
	v3_dataframe, _ = make_distance_dataframe(_400_or_less_images, v3_feature)
	color_dataframe, _ = make_distance_dataframe(_400_or_less_images, color_feature)
	ava_dataframe, ava_dict = make_distance_dataframe(_400_or_less_images, ava_score)
	
	# mix all the types of distance dataframe
	mixed_dataframe = v3_dataframe.add(FLAGS.mix_rate1*color_dataframe)
	mixed_dataframe = mixed_dataframe.add(FLAGS.mix_rate2*ava_dataframe)
	end_time = time.time()
	print("filtering images: " + str(end_time-start_time) + 'seconds')

	target_image = min(ava_dict, key=ava_dict.get) # first image should be one that has highest AVA score
	arrangement_id_list = dist_ordered_list_of_id(mixed_dataframe, target_image) # make a ordere list of images
	visualize_result(arrangement_id_list) # visualize the result

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--feature_dir',
		type=str,
		default='',
		help="""\
		feature directory path\
		"""
	)

	parser.add_argument(
		'--image_dir',
		type=str,
		default='',
		help="""\
		image directory path\
		"""
	)

	parser.add_argument(
		'--model_dir',
		type=str,
		default=os.getcwd()+'/AVA_retrain_data/imagenet',
		help="""\
		it is directory path of mode inception graph\
		"""
	)

	parser.add_argument(
		'--num_cluster',
		type=int,
		default=400,
		help="""\
		it is number of cluster\
		"""
	)

	parser.add_argument(
		'--tolerance',
		type=float,
		default=0.00001,
		help="""\
		it is relative tolerance of kmeans\
		"""
	)

	parser.add_argument(
		'--mix_rate1',
		type=int,
		default=1,
	)

	parser.add_argument(
		'--mix_rate2',
		type=int,
		default=1,
	)

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
