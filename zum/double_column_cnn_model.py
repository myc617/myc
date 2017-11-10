from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile
import csv

import pandas as pd
import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

MODEL_INPUT_WIDTH = 224
MODEL_INPUT_HEIGHT = 224
MODEL_INPUT_DEPTH = 3
MAX_NUM_IMAGES_PER_CATEGORY = 2 ** 27 - 1 #~134M

FLAGS = None

def create_image_lists(image_dir, testing_percentage, validation_percentage):

	if not gfile.Exists(image_dir):
		print("Image director '" + image_dir + "' not found.")
		return None
	result = {}
	extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
	file_list = []
	for extension in extensions:
		file_glob = os.path.join(image_dir, '*.' + extension)
		file_list.extend(gfile.Glob(file_glob))
		if not file_list:
			print('No files found')
			continue
	if len(file_list) < 20000:
		print('WARNING: Folder has less than 20000 images, which may cause issues.')
	elif len(file_list) > MAX_NUM_IMAGES_PER_CATEGORY:
		print('WARNING: Folder {} has more than {} images. Some images will '
				'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CATEGORY))
	training_images = []
	testing_images = []
	validation_images = []
	for file_name in file_list:
		base_name = os.path.basename(file_name)
		hash_name = re.sub(r'_nohash_.*$', '', file_name)
		hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
		percentage_hash = ((int(hash_name_hashed, 16) %
							(MAX_NUM_IMAGES_PER_CATEGORY + 1)) *
							(100.0 / MAX_NUM_IMAGES_PER_CATEGORY))
		if percentage_hash < validation_percentage:
			validation_images.append(base_name)
		elif percentage_hash < (testing_percentage + validation_percentage):
			testing_images.append(base_name)
		else:
			training_images.append(base_name)
	result = {
		'training': training_images,
		'testing': testing_images,
		'validation': validation_images
	}
	return result

def create_ava_score_dict(ava_dir):
	
	if not os.path.exists(ava_dir):
		return None
	result = {}
	ava_dir_string = str(ava_dir)
	ava_scores = open(ava_dir_string)
	ava_scores = pd.read_csv(ava_scores, delim_whitespace = True, header = None)
	image_numbers = ava_scores[0]	
	i = 0
	for image_number in image_numbers:
		ava_score_list = []
		ava_score = ava_scores.loc[ava_scores[0] == image_number]
		ava_index = ava_score.index[0]

		mean = ava_score.get_value(index = ava_index, col = 14)
		ava_score_list.append(mean)
		glob_bin = ava_score.get_value(index = ava_index, col = 15)
		ava_score_list.append(glob_bin)
		chal_bin = ava_score.get_value(index = ava_index, col = 16)
		ava_score_list.append(chal_bin)
		result[image_number] = ava_score_list
		i += 1
		if (i%10000) == 0:
			print(i)

	return result

def get_image_path(image_lists, index, image_dir, category):
	
	if category not in image_lists:
		tf.logging.fatal('Category does not exist %s.', category)
	category_list = image_lists[category]
	if not category_list:
		tf.logging.fatal('There are no images in the cateogry %s.', category)
	mod_index = index % len(category_list)
	base_name = category_list[mod_index]
	full_path = os.path.join(image_dir, base_name)
	return full_path

def rescale_image():
	
	jpeg_data = tf.placeholder(tf.string, name='ResizeJPGInput')
	decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
	decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
	decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
	model_width_value = tf.constant(MODEL_INPUT_WIDTH)
	model_height_value = tf.constant(MODEL_INPUT_HEIGHT)
	model_rescale_shape = tf.stack([model_height_value, model_width_value])
	rescaled_image = tf.image.resize_bilinear(decoded_image_4d,
											model_rescale_shape)
	rescaled_image = tf.reshape(rescaled_image, [MODEL_INPUT_HEIGHT, 
								MODEL_INPUT_WIDTH, MODEL_INPUT_DEPTH])
	return jpeg_data, rescaled_image

def center_crop_or_pad_image():
	
	jpeg_data = tf.placeholder(tf.string, name='CenterCropJPGInput')
	decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
	decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
	decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
	center_cropped_image = tf.image.resize_image_with_crop_or_pad(decoded_image_4d,
													target_height=MODEL_INPUT_HEIGHT,
													target_width=MODEL_INPUT_WIDTH)
	center_cropped_image = tf.reshape(center_cropped_image, [MODEL_INPUT_HEIGHT,
										MODEL_INPUT_WIDTH, MODEL_INPUT_DEPTH])
	return jpeg_data, center_cropped_image

def get_random_resized_input_tensor(sess, image_lists, how_many, category, image_dir,
									rescaling_input_tensor, rescaling_output_tensor,
									center_cropping_input_tensor, center_cropping_output_tensor,
									ava_scores):

	rescale_inputs = []
	center_crop_inputs = []
	mean_ground_truths = []
	glob_bin_ground_truths = []
	chal_bin_ground_truths = []
	filenames = []
	is_testing_set = False

	if how_many < 0:
		how_many = len(image_lists[category])
		is_testing_set = True

	for unused_i in range(how_many):
		if is_testing_set:
			image_index = unused_i
		else:
			image_index = random.randrange(MAX_NUM_IMAGES_PER_CATEGORY + 1)
		image_path = get_image_path(image_lists, image_index, image_dir, category)
		if not gfile.Exists(image_path):
			tf.logging.fatal('File does not exist %s', image_path)
		jpeg_data = gfile.FastGFile(image_path, 'rb').read()
		
		rescale_input = sess.run(rescaling_output_tensor,
								{rescaling_input_tensor: jpeg_data})
		center_crop_input = sess.run(center_cropping_output_tensor,
									{center_cropping_input_tensor: jpeg_data})
		
		mean_ground_truth = (get_ground_truth(image_path, ava_scores)[0] - 5) / 5
		glob_bin_ground_truth = get_ground_truth(image_path, ava_scores)[1] * 2 - 1
		chal_bin_ground_truth = get_ground_truth(image_path, ava_scores)[2] * 2 - 1

		rescale_inputs.append(rescale_input)
		center_crop_inputs.append(center_crop_input)
		mean_ground_truths.append(mean_ground_truth)
		glob_bin_ground_truths.append(glob_bin_ground_truth)
		chal_bin_ground_truths.append(chal_bin_ground_truth)
		filenames.append(image_path)

	rescale_inputs_array = np.array(rescale_inputs)
	center_crop_inputs_array = np.array(center_crop_inputs)

	return (rescale_inputs_array, center_crop_inputs_array, mean_ground_truths,
			glob_bin_ground_truths, chal_bin_ground_truths, filenames)

def get_ground_truth(image_path, ava_scores):

	image_name = image_path.split('/')[-1]
	image_name = image_name.split('.')[0]
	image_name = int(image_name)
	
	ground_truth = ava_scores[image_name]
	return ground_truth

def batch_norm(x, n_out):

	with tf.variable_scope('batch_norm') as scope:
		beta = tf.Variable(tf.constant(0.0, shape = [n_out]), name = 'beta',
										trainable = True)
		gamma = tf.Variable(tf.constant(1.0, shape = [n_out]), name = 'gamma',
										trainable = True)
		batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name = 'moments')
		ema = tf.train.ExponentialMovingAverage(decay = 0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = mean_var_with_update()
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed

def variable_summaries(var):

	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)

def build_convolution_layers():

	# image input
	# ground truth
	with tf.name_scope('input'):
		image_input = tf.placeholder(tf.float32,
			[None, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_DEPTH],
			name = 'ImageInputPlaceholder')

		m_ground_truth_input = tf.placeholder(tf.float32, [None],
			name = 'MeanGroundTruthInput')
		g_ground_truth_input = tf.placeholder(tf.float32, [None],
			name = 'GlobBinGroundTruthInput')
		c_ground_truth_input = tf.placeholder(tf.float32, [None],
			name = 'ChalBinGroundTruthInput')

	# conv1_1
	with tf.name_scope('conv1_1') as scope:
		kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype = tf.float32,
												stddev = np.sqrt(2 / (11*11*3 + 11*11*64))),
												name = 'weights')
		variable_summaries(kernel)
		conv = tf.nn.conv2d(image_input, kernel, [1, 2, 2, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
							trainable = True, name = 'biases')
		variable_summaries(biases)

		out = tf.nn.bias_add(conv, biases)
		tf.summary.histogram('pre_activations', out)
		bn = batch_norm(out, 64)
		conv1_1 = tf.nn.relu(bn, name = scope)
		tf.summary.histogram('activations', conv1_1)
	
	# pool1
	pool1 = tf.nn.max_pool(conv1_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], 
							padding = 'SAME', name = 'pool1')

	# conv2_1
	with tf.name_scope('conv2_1') as scope:
		kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 64], dtype = tf.float32,
												stddev = np.sqrt(2 / (5*5*64*2)),
												name = 'weights'))
		variable_summaries(kernel)
		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding = 'SAME')
		biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
							trainable = True, name = 'biases')
		variable_summaries(biases)

		out = tf.nn.bias_add(conv, biases)
		tf.summary.histogram('pre_activations', out)
		bn = batch_norm(out, 64)
		conv2_1 = tf.nn.relu(bn, name = scope)
		tf.summary.histogram('activations', conv2_1)
	
	# pool2
	pool2 = tf.nn.max_pool(conv2_1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
							padding = 'SAME', name = 'pool2')

	# conv3_1
	with tf.name_scope('conv3_1') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype = tf.float32,
												stddev = np.sqrt(2 / (3*3*64*2)),
												name = 'weights'))
		variable_summaries(kernel)
		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding = 'SAME')
		biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
							trainable = True, name = 'biases')
		variable_summaries(biases)

		out = tf.nn.bias_add(conv, biases)
		tf.summary.histogram('pre_activatons', out)
		conv3_1 = tf.nn.relu(out, name = scope)
		tf.summary.histogram('activations', conv3_1)

	# conv3_2
	with tf.name_scope('conv3_2') as scope:
		kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype = tf.float32,
												stddev = np.sqrt(2 / (3*3*64*2)),
												name = 'weights'))
		variable_summaries(kernel)
		conv = tf.nn.conv2d(conv3_1, kernel, [1, 1, 1, 1], padding = 'SAME')
		biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32),
							trainable = True, name = 'biases')
		variable_summaries(biases)

		out = tf.nn.bias_add(conv, biases)
		tf.summary.histogram('pre_activations', out)
		conv3_2 = tf.nn.relu(out, name = scope)
		tf.summary.histogram('activations', conv3_2)

	# fc0
	with tf.name_scope('fc0') as scope:
		shape = int(np.prod(conv3_2.get_shape()[1:]))
		fc0w = tf.Variable(tf.truncated_normal([shape, 1000],
													dtype = tf.float32,
													stddev = 0.00002), name = 'weights')
		variable_summaries(fc0w)
		fc0b = tf.Variable(tf.constant(1.0, shape = [1000], dtype = tf.float32),
							trainable = True, name = 'biases')
		variable_summaries(fc0b)

		conv3_2_flat = tf.reshape(conv3_2, [-1, shape])

		fc0l = tf.nn.bias_add(tf.matmul(conv3_2_flat, fc0w), fc0b)
		tf.summary.histogram('pre_activations', fc0l)
		fc0 = tf.nn.relu(fc0l)
		tf.summary.histogram('activatons', fc0)
	
	# fc1
	with tf.name_scope('fc1') as scope:
		fc1w = tf.Variable(tf.truncated_normal([1000, 256], dtype = tf.float32,
												stddev = np.sqrt(2 / (512*2)),
												name = 'weights'))
		variable_summaries(fc1w)
		fc1b = tf.Variable(tf.constant(1.0, shape = [256], dtype = tf.float32),
							trainable = True, name = 'biases')
		variable_summaries(fc1b)

		fc1l = tf.nn.bias_add(tf.matmul(fc0, fc1w), fc1b)
		tf.summary.histogram('pre_activations', fc1l)
		fc1 = tf.nn.relu(fc1l)
		tf.summary.histogram('activations', fc1)

	# mean
	with tf.name_scope('mean_prediction') as scope:
		mean_w = tf.Variable(tf.truncated_normal([256, 1], dtype = tf.float32,
												stddev = np.sqrt(2 / (256 + 1)),
												name = 'weights'))
		variable_summaries(mean_w)
		mean_b = tf.Variable(tf.constant(0.0, shape = [1], dtype = tf.float32),
								trainable = True, name = 'biases')
		variable_summaries(mean_b)
		
		mean_l = tf.nn.bias_add(tf.matmul(fc1, mean_w), mean_b)
		tf.summary.histogram('pre_activations', mean_l)
		mean = tf.tanh(mean_l)
		tf.summary.histogram('activations', mean)

	# glob_bin
	with tf.name_scope('glob_bin_prediction') as scope:
		glob_bin_w = tf.Variable(tf.truncated_normal([256, 1], dtype = tf.float32,
													stddev = np.sqrt(2 / (256 + 1)),
													name = 'weights'))
		variable_summaries(glob_bin_w)
		glob_bin_b = tf.Variable(tf.constant(0.0, shape = [1], dtype = tf.float32),
									trainable = True, name = 'biases')
		variable_summaries(glob_bin_b)

		glob_bin_l = tf.nn.bias_add(tf.matmul(fc1, glob_bin_w), glob_bin_b)
		tf.summary.histogram('pre_activations', glob_bin_l)
		glob_bin = tf.tanh(glob_bin_l)
		tf.summary.histogram('activations', glob_bin)

	# chal_bin
	with tf.name_scope('chal_bin_prediction') as scope:
		chal_bin_w = tf.Variable(tf.truncated_normal([256, 1], dtype = tf.float32,
													stddev = np.sqrt(2 / (256 + 1)),
													name = 'weights'))
		variable_summaries(chal_bin_w)
		chal_bin_b = tf.Variable(tf.constant(0.0, shape = [1], dtype = tf.float32),
									trainable = True, name = 'biases')
		variable_summaries(chal_bin_b)

		chal_bin_l = tf.nn.bias_add(tf.matmul(fc1, chal_bin_w), chal_bin_b)
		tf.summary.histogram('pre_activations', chal_bin_l)
		chal_bin = tf.tanh(chal_bin_l)
		tf.summary.histogram('activations', chal_bin)

	# cost
	with tf.name_scope('cost') as scope:
		mean_cost = tf.reduce_mean(tf.square(mean - m_ground_truth_input))
		tf.summary.scalar('mean_cost', mean_cost)

		glob_bin_cost = tf.reduce_mean(tf.square(glob_bin - g_ground_truth_input))
		tf.summary.scalar('glob_bin_cost', glob_bin_cost)

		chal_bin_cost = tf.reduce_mean(tf.square(chal_bin - c_ground_truth_input))
		tf.summary.scalar('chal_bin_cost', chal_bin_cost)

		# total_cost = mean_cost + glob_bin_cost + chal_bin_cost
		total_cost = glob_bin_cost
		tf.summary.scalar('total_cost', total_cost)

	# train
	with tf.name_scope('train') as scope:
		# learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, FLAGS.how_many_training_steps,
		# 											250, 0.8, staircase = True)
		optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
		# gradients, variables = zip(*optimizer.coumpute_gradients(total_cost))
		# gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		# train_step = optimizer.apply_gradients(zip(gradients, variables))
		train_step = optimizer.minimize(total_cost)

	return (train_step, total_cost, image_input, m_ground_truth_input, g_ground_truth_input,
			c_ground_truth_input, mean, glob_bin, chal_bin, mean_cost, glob_bin_cost,
			chal_bin_cost)

def build_evaluation_layers(mean, glob_bin, chal_bin, m_ground_truth_input, 
							g_ground_truth_input, c_ground_truth_input):

	with tf.name_scope('mean_accuracy') as scope:
		m_prediction = tf.square(mean - m_ground_truth_input)
		m_evaluation_step = tf.reduce_mean(tf.cast(m_prediction, tf.float32))
		tf.summary.scalar('accuracy', m_evaluation_step)

	with tf.name_scope('glob_bin_accuracy') as scope:
		g_prediction = tf.greater(glob_bin, 0.0)
		g_prediction_result = tf.equal(g_prediction, tf.equal(g_ground_truth_input, 1.0))
		g_evaluation_step = tf.reduce_mean(tf.cast(g_prediction_result, tf.float32))
		tf.summary.scalar('accuracy', g_evaluation_step)

	with tf.name_scope('chal_bin_accuracy') as scope:
		c_prediction = tf.greater(chal_bin, 0.0)
		c_prediction_result = tf.equal(c_prediction, tf.equal(c_ground_truth_input, 1.0))
		c_evaluation_step = tf.reduce_mean(tf.cast(c_prediction_result, tf.float32))
		tf.summary.scalar('accuracy', c_evaluation_step)

	return (m_evaluation_step, g_evaluation_step, c_evaluation_step,
			m_prediction, g_prediction, c_prediction)

def main(_):

	if tf.gfile.Exists(FLAGS.summaries_dir):
		tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
	tf.gfile.MakeDirs(FLAGS.summaries_dir)

	image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
									FLAGS.validation_percentage)
	ava_scores = create_ava_score_dict(FLAGS.ava_dir)

	with tf.Session() as sess:
		
		(rescaled_input_tensor, rescaled_image) = rescale_image()
		(center_cropped_input_tensor, center_cropped_image) = center_crop_or_pad_image()

		(train_step, total_cost, image_input, m_ground_truth_input, g_ground_truth_input,
		c_ground_truth_input, mean, glob_bin, chal_bin, mean_cost, glob_bin_cost,
		chal_bin_cost) = build_convolution_layers()

		(m_evaluation_step, g_evaluation_step, c_evaluation_step,
		m_prediction, g_prediction, c_prediction) = build_evaluation_layers(mean, glob_bin,
				chal_bin, m_ground_truth_input, g_ground_truth_input, c_ground_truth_input)

		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
				sess.graph)
		validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

		init = tf.global_variables_initializer()
		sess.run(init)

		f = open('decay_test(0.0001,0.7,1000).txt', 'w')

		for i in range(FLAGS.how_many_training_steps):
			(train_image_input, _, 
			train_m_ground_truth_input, train_g_ground_truth_input,
			train_c_ground_truth_input, _) = get_random_resized_input_tensor(
				sess, image_lists, FLAGS.train_batch_size, 'training',
				FLAGS.image_dir, rescaled_input_tensor, rescaled_image,
				center_cropped_input_tensor, center_cropped_image,
				ava_scores)

			train_summary, _ = sess.run(
				[merged, train_step],
				feed_dict = {image_input: train_image_input,
							m_ground_truth_input: train_m_ground_truth_input,
							g_ground_truth_input: train_g_ground_truth_input,
							c_ground_truth_input: train_c_ground_truth_input})
			train_writer.add_summary(train_summary, i)

			is_last_step = (i + 1 == FLAGS.how_many_training_steps)
			if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
				(m_train_accuracy, g_train_accuracy, c_train_accuracy, total_cost_print, 
				mean_cost_print, glob_bin_cost_print, chal_bin_cost_print, glob_bin_print) = sess.run(
					[m_evaluation_step, g_evaluation_step, c_evaluation_step,
					total_cost, mean_cost, glob_bin_cost, chal_bin_cost, glob_bin],
					feed_dict={image_input: train_image_input,
								m_ground_truth_input: train_m_ground_truth_input,
								g_ground_truth_input: train_g_ground_truth_input,
								c_ground_truth_input: train_c_ground_truth_input})
				print('%s: Step %d:' % (datetime.now(), i))
				print('Mean train accuracy(MSE) = %2f' % (m_train_accuracy))
				print('Glob bin train accuracy = %.1f%%' % (g_train_accuracy * 100))
				print('Chal bin train accuracy = %.1f%%' % (c_train_accuracy * 100))
				print('Total cost = %.3f' % (total_cost_print))
				print('Mean cost = %.3f' % (mean_cost_print))
				print('Glob bin cost = %.3f' % (glob_bin_cost_print))
				print('Chal bin cost = %.3f' % (chal_bin_cost_print))
				print('glob bin predictions = ')
				print(glob_bin_print)

				print('%s: Step %d:' % (datetime.now(), i), file=f)
				print('Mean train accuracy(MSE) = %2f' % (m_train_accuracy), file=f)
				print('Glob bin train accuracy = %.1f%%' % (g_train_accuracy * 100), file=f)
				print('Chal bin train accuracy = %.1f%%' % (c_train_accuracy * 100), file=f)
				print('Total cost = %.3f' % (total_cost_print), file=f)
				print('Mean cost = %.3f' % (mean_cost_print), file=f)
				print('Glob bin cost = %.3f' % (glob_bin_cost_print), file=f)
				print('Chal bin cost = %.3f' % (chal_bin_cost_print), file=f)
				print('glob bin predictions = ', file=f)
				print(glob_bin_print, file=f)

				(validation_image_input, _,
				validation_m_ground_truth_input, validation_g_ground_truth_input,
				validation_c_ground_truth_input, _) = get_random_resized_input_tensor(
					sess, image_lists, FLAGS.validation_batch_size, 'validation',
					FLAGS.image_dir, rescaled_input_tensor, rescaled_image,
					center_cropped_input_tensor, center_cropped_image,
					ava_scores)

				(validation_summary, m_validation_accuracy, g_validation_accuracy,
				c_validation_accuracy) = sess.run(
					[merged, m_evaluation_step, g_evaluation_step, c_evaluation_step],
					feed_dict={image_input: validation_image_input,
								m_ground_truth_input: validation_m_ground_truth_input,
								g_ground_truth_input: validation_g_ground_truth_input,
								c_ground_truth_input: validation_c_ground_truth_input})
				validation_writer.add_summary(validation_summary, i)
				print('%s: Step %d:' % (datetime.now(), i))
				print('Mean validation accuracy(MSE) = %.2f' % (m_validation_accuracy))
				print('Glob bin validation accuracy = %.1f%%' % (g_validation_accuracy * 100))
				print('Chal bin validation accuracy = %.1f%%' % (c_validation_accuracy * 100))
				print('(N=%d)' % (len(validation_image_input)))

				print('%s: Step %d:' % (datetime.now(), i), file=f)
				print('Mean validation accuracy(MSE) = %.2f' % (m_validation_accuracy), file=f)
				print('Glob bin validation accuracy = %.1f%%' % (g_validation_accuracy * 100), file=f)
				print('Chal bin validation accuracy = %.1f%%' % (c_validation_accuracy * 100), file=f)
				print('(N=%d)' % (len(validation_image_input)), file=f)

		(test_image_input, _,
		test_m_ground_truth_input, test_g_ground_truth_input,
		test_c_ground_truth_input, test_filenames) = get_random_resized_input_tensor(
			sess, image_lists, FLAGS.test_batch_size, 'testing',
			FLAGS.image_dir, rescaled_input_tensor, rescaled_image,
			center_cropped_input_tensor, center_cropped_image,
			ava_scores)

		(m_test_accuracy, g_test_accuracy, c_test_accuracy,
		m_predictions, g_predictions, c_predictions) = sess.run(
			[m_evaluation_step, g_evaluation_step, c_evaluation_step,
			m_prediction, g_prediction, c_prediction],
			feed_dict={image_input: test_image_input,
						m_ground_truth_input: test_m_ground_truth_input,
						g_ground_truth_input: test_g_ground_truth_input,
						c_ground_truth_input: test_c_ground_truth_input})
		print('Final mean test accuracy(MSE) = %.2f' % (m_test_accuracy * 100))
		print('Final glob bin test accuracy = %.2f%%' % (g_test_accuracy * 100))
		print('Final chal bin test accuracy = %.2f%%' % (c_test_accuracy * 100))
		print('(N=%d)' % (len(test_image_input)))

		print('Final mean test accuracy(MSE) = %.2f' % (m_test_accuracy * 100), file=f)
		print('Final glob bin test accuracy = %.2f%%' % (g_test_accuracy * 100), file=f)
		print('Final chal bin test accuracy = %.2f%%' % (c_test_accuracy * 100), file=f)
		print('(N=%d)' % (len(test_image_input)), file=f)

		if FLAGS.print_isclassified_test_images:
			print('=== MISCLASSIFIED TEST IMAGES ===')
			for i, test_filename in enumerate(test_filenames):
				if g_predictions[i] != (test_g_ground_truth_input[i]==1.0):
					print('Glob bin: \n %70s' % (test_filename))
				if c_predictions[i] != (test_c_ground_truth_input[i]==1.0):
					print('Chal bin: \n %70s' % (test_filename))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--image_dir',
		type=str,
		default='',
		help='Path to folder of labeled images.'
	)

	parser.add_argument(
		'--ava_dir',
		type=str,
		default='',
		help='Path to folder of AVA.txt.'
	)

	parser.add_argument(
		'--summaries_dir',
		type=str,
		default = os.getcwd() + '/DCNN_AVA_data/retrain_logs',
		help='Where to save summary logs for TensorBoard.'
	)

	parser.add_argument(
		'--how_many_training_steps',
		type=int,
		default=6400,
		help='How many training steps to run before ending.'
	)

	parser.add_argument(
		'--learning_rate',
		type=float,
		default=0.00001,
		help='How large a learning rate to use when training.'
	)

	parser.add_argument(
		'--testing_percentage',
		type=int,
		default=10,
		help='What percentage of images to use as a test set.'
	)

	parser.add_argument(
		'--validation_percentage',
		type=int,
		default=10,
		help='What percentage of images to use as a validation set.'
	)

	parser.add_argument(
		'--eval_step_interval',
		type=int,
		default=10,
		help='How often to evaluate the training results.'
	)
  
	parser.add_argument(
		'--train_batch_size',
		type=int,
		default=64,
		help='How many images to train on at a time.'
	)
  
	parser.add_argument(
		'--test_batch_size',
		type=int,
		default=64,
		help="""\
		How many images to test on. This test set is only used once, to evaluate
		the final accuracy of the model after training completes.
		A value of -1 causes the entire test set to be used, which leads to more
		stable results across runs.\
		"""
	)

	parser.add_argument(
		'--validation_batch_size',
		type=int,
		default=64,
		help="""\
		How many images to use in an evaluation batch. This validation set is
		used much more often than the test set, and is an early indicator of how
		accurate the model is during training.
		A value of -1 causes the entire validation set to be used, which leads to
		more stable results across training iterations, but may be slower on large
		training sets.\
		"""
	)

	parser.add_argument(
		'--print_misclassified_test_images',
		default=False,
		help="""\
		Whether to print out a list of all misclassified test images.\
		""",
		action='store_true'
	)

	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
