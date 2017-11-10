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
import numpy as np
import scipy
import scipy.spatial.distance
import scipy.stats
from pandas import Series, DataFrame
import pandas as pd
import tensorflow as tf
from six.moves import urllib
from colorthief import ColorThief
import time

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_CHANNEL = 3

BOTTLENECK_TENSOR = 'pool_3/_reshape:0'
RESIZED_INPUT_TENSOR = 'ResizeBilinear:0'
TRAINING_BATCH_OPTION = 'batch_normalization/keras_learning_phase:0'
MEAN_SCORE_TENSOR = 'mean_output/Tanh:0'
RESCALED_INPUT_TENSOR = 'input_1:0'
CENTERED_INPUT_TENSOR = 'input_2:0'

FLAGS = None

def inception_v3_download():
	"""
	Download and extract inception v3 model.
	"""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully download', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def create_inception_v3_graph():
	"""
	creates a graph from saved inception v3 GraphDef file and returns a Graph object.
	
	Returns:
		inception v3 Graph object (to extract 2048 bottleneck features)
	"""
    with tf.Graph().as_default() as graph:
        model_filename = os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR, RESIZED_INPUT_TENSOR])
                                      
    return graph, bottleneck_tensor, resized_input_tensor

def create_ava_inception_v3_graph():
	"""
	creates a graph from saved AVA dataset trained inception v3 GraphDef file and returns a Graph object.
	(GraphDef file must be placed at model directory)

	Reutrns:
		AVA dataset trained inception v3 Graph object (to estimate AVA score)
	"""
    with tf.Graph().as_default() as graph:
        model_filename = 'AVA_estimator.pb'
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            mean_score_tensor, rescaled_input_tensor, centered_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                                    MEAN_SCORE_TENSOR, RESCALED_INPUT_TENSOR,
                                    CENTERED_INPUT_TENSOR]))

	return graph, mean_score_tensor, rescaled_input_tensor, centered_input_tensor

def rescaled_image():
	"""
	makes a rescaling tensor which makes 299x299 rescaled image data from input image data

	Returns:
		jpeg_data: input tensor which takes image data for rescaling
		rescaled_image: output tensor which makes 299x299 rescaled image data
	"""
    jpeg_data = tf.placeholder(tf.string, name = 'ResizeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_CHANNEL)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    model_width_value = tf.constant(MODEL_INPUT_WIDTH)
    model_height_value = tf.constant(MODEL_INPUT_HEIGHT)
    model_rescale_shape = tf.stack([model_width_value, model_height_value])
    rescaled_image = tf.image.resize_bilinear(decoded_image_4d, model_rescale_shape)
        
    return jpeg_data, rescaled_image

def centered_image():
	"""
	makes a center cropping tensor which makes 299x299 center cropped image data from input image data

	Returns:
		jpeg_data: input tensor which takes image data for center cropping
		centered_image: output tensor which makes 299x299 center cropped image data
	"""
	jpeg_data = tf.placeholder(tf.string, name = 'CenteredJPGInput')
	decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_CHANNEL)
	decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
	decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
	centered_image = tf.image.resize_image_with_crop_or_pad(decoded_image_4d,
											target_height=MODEL_INPUT_HEIGHT,
											target_width=MODEL_INPUT_WIDTH)

	return jpeg_data, centered_image

def get_bottleneck_feature(sess, image_dir, image_id, resizing_input_tensor,
        resizing_output_tensor, image_input_tensor, bottleneck_tensor):
	"""
	gets 2048 dimensions feature vector from inception v3 model

	Args:
		sess: inception v3 session object
		image_dir: directory where the images are placed
		image_id: name of image which will extract feature vector
		resizing_input_tensor: input tensor which takes image data for rescaling
		resizing_output_tensor: output tensor which makes 299x299 rescaled image data
		image_input_tensor: input tensor which takes image data for feature extraction
		bottleneck_tensor: output tensor which makes 2048 dimention vector

	Returns:
		bottleneck_value: 2048 dimention feature vector
	"""
    image_path = os.path.join(image_dir, image_id)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    resized_image_data = sess.run(resizing_output_tensor,
        {resizing_input_tensor: jpeg_data})
    bottleneck = sess.run(bottleneck_tensor,
        {image_input_tensor: resized_image_data})
    bottleneck_value = np.squeeze(bottleneck)

    return bottleneck_value

def get_ava_score(sess, image_dir, image_id, rescaling_input_tensor, rescaling_output_tensor,
                  centering_input_tensor, centering_output_tensor, rescaled_input_tensor,
                  centered_input_tensor, ava_score_tensor):
	"""
	get AVA score of image estimated by AVA trained inception v3 model

	Args:
		sess: inception v3 session object
		image_dir: directory where the images are placed
		image_id: name of image which will extract feature vector
		rescaling_input_tensor: input tensor which takes image data for rescaling
		rescaling_output_tensor: output tensor which makes 299x299 rescaled image data
		centering_input_tensor: input tensor which takes image data for center cropping
		centering_output_tensor: output tensor which makes 299x299 center cropped image data
		rescaled_input_tensor: input tensor which takes rescaled image data for feature extraction
		centered_input_tensor: input tensor which takes center cropped image data for feature extraction
		ava_score_tensor: output tensor which estimate AVA score of input image

	Returns:
		ava_score_value: AVA score of input image
	"""
    image_path = os.path.join(image_dir, image_id)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    rescaled_image_data = sess.run(rescaling_output_tensor,
                                  {rescaling_input_tensor: jpeg_data})
    centered_image_data = sess.run(centering_output_tensor,
                                   {centering_input_tensor: jpeg_data})
    ava_score = sess.run(ava_score_tensor,
                         {rescaled_input_tensor: rescaled_image_data,
                         centered_input_tensor: centered_image_data,
						 TRAINING_BATCH_OPTION: False})
    ava_score_value = np.squeeze(ava_score)

    return ava_score_value

def make_image_list(image_dir):
	"""
	makes list of names of images which is placed at image_dir directory

	Args:
		image_dir: directory of target images

	Returns:
		image_list: list of names of target images
	"""
    if not os.path.exists(image_dir):
        print("There is no " + image_dir + " directory")
        return None
    image_list = os.listdir(image_dir)
        
    return image_list

def main(_):
	features_of_images = {}

	inception_v3_download()
	v3_graph, bottleneck_tensor, image_input_tensor = create_inception_v3_graph()
	ava_graph, ava_output, rescaled_input, centered_input = create_ava_inception_v3_graph()
	image_list = make_image_list(FLAGS.image_dir)

	# makes dictionary which key is name of image and value is 2048 feature vector
	start_time = time.time()
	with tf.Session(graph=v3_graph) as sess:
		resizing_input_tensor, resizing_output_tensor = rescaled_image()
		for image_id in image_list:
			image = get_bottleneck_feature(sess, FLAGS.image_dir, image_id,
				resizing_input_tensor, resizing_output_tensor, image_input_tensor,
				bottleneck_tensor)
			features_of_images[image_id] = [image]
	end_time = time.time()
	print("v3 feature extraction: " + str(end_time - start_time) + "seconds")

	# append color feature vector at each dictionary's value
	start_time = time.time()
	for image_id in image_list:
		image_path = os.path.join(FLAGS.image_dir, image_id)
		image = ColorThief(image_path)
		dominant_color = list(image.get_color(quality=5))
		features_of_images[image_id].append(dominant_color)
	end_time = time.time()
	print("color feature extraction: " + str(end_time - start_time) + "seconds")

	# append AVA score value at each dictionary's value
	start_time = time.time()
	with tf.Session(graph=ava_graph) as sess:
		rescaling_input_tensor, rescaling_output_tensor = rescaled_image()
		centering_input_tensor, centering_output_tensor = centered_image()
		for image_id in image_list:
			image = get_ava_score(sess, FLAGS.image_dir, image_id,
				rescaling_input_tensor, rescaling_output_tensor,
				centering_input_tensor, centering_output_tensor,
				rescaled_input, centered_input, ava_output)
			ava_score = (image+1)*5
			ava_score = 1 / ava_score
			features_of_images[image_id].append([float(ava_score)])
	end_time = time.time()
	print("ava score extraction: " + str(end_time - start_time) + "seconds")

	# make 2048 size zero vector
	# it is necessary to make all different features have same length
	zero_list = []
	for i in range(BOTTLENECK_TENSOR_SIZE):
		zero_list.append(0)

	# make a feature dictionary and put every feature of one image
	# save the dictionary as csv file and do it iteratively
	for key in features_of_images.keys():
		image_features = {}
		features_of_images[key][1].extend(zero_list[0:-3])
		features_of_images[key][2].extend(zero_list[0:-1])
		image_features['v3_feature'] = features_of_images[key][0]
		image_features['color_feature'] = features_of_images[key][1]
		image_features['ava_score'] = features_of_images[key][2]

		image_dataframe = DataFrame(image_features)
		dir_name = FLAGS.image_dir.split('/')[0] + '_image_features'
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
		name = dir_name + '/' + key.split('.')[0] + '.csv'
		image_dataframe.to_csv(name)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

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

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
