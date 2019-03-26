# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:12:56 2019

@author: Lynden
"""
import tensorflow as tf
import cv2
import os
#import matplotlib.pyplot as plt

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def preprocess(imageRawDir, imageDir):
    """
    images preprocess
    Arguments:
        imageRawDir -- directory of primary images.
        imageDir -- directory of processed images.
    Return: none.
    """
    imageNames = os.listdir(imageRawDir)
    #print(imageNames)
    for index, imageName in enumerate(imageNames):
        image = cv2.imread(os.path.join(imageRawDir, imageName))
        image = cv2.resize(image, (256, 256))
        savePath = os.path.join(imageDir, imageName)
        cv2.imwrite(savePath, image)
        

def createTFRecord(imageDir):
    # create a writer to write TFRecord file
    writer = tf.python_io.TFRecordWriter(os.path.join(imageDir, "train.tfrecords"))
    
    # label cat as 0; dog as 1
    csv = [(os.path.join(imageDir, 'cat.0.jpg'), 0), (os.path.join(imageDir, 'cat.1.jpg'), 0), (os.path.join(imageDir, 'cat.2.jpg'), 0),
           (os.path.join(imageDir, 'dog.0.jpg'), 1), (os.path.join(imageDir, 'dog.1.jpg'), 1), (os.path.join(imageDir, 'dog.2.jpg'), 1)]
    
    for path, lab in csv:
        img = cv2.imread(path)
        # ==== encode the .jpg image into .jpeg image format === #
        # ==== Tensorflow support (JPEG, PNG or GIF) formats ==== #
        img_raw = cv2.imencode('.jpeg', img)[1].tostring()
        
        # write image data(pixel values and label) to Example Protocol Buffer
        example = tf.train.Example(features = tf.train.Features(feature = {
            "name": _bytes_feature(str(path).encode('utf-8')),
            "label": _int64_feature(lab),
            "image_raw": _bytes_feature(img_raw),
            }))
    
        # write an example to TFRecord file
        writer.write(example.SerializeToString())
    
    writer.close()


def parse_example_proto(example_serialized):
    # parse features from the serialized data
    
    # Create a description of the features.  
    feature_map = {
            'name': tf.FixedLenFeature([], tf.string, default_value=''),
            'image_raw': tf.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.FixedLenFeature([], tf.int64, default_value=-1)
    }
    
    # Parse the input tf.Example proto using the dictionary above.
    features = tf.parse_single_example(example_serialized, feature_map)
    print('parse feature: {}'.format(features))
    
    # obtain the label and bounding boxes
    label = tf.cast(features['label'], dtype=tf.int32)
    
    return features['name'], features['image_raw'], label


def preprocess_image(image_buffer, is_training=False):
    if is_training:
        # For training, we want to do somethings.
        image = tf.image.decode_jpeg(image_buffer, channels=3)
    else:
        # For validation, we want to do somethings.
        image = tf.image.decode_jpeg(image_buffer, channels=3)
    
    return image

def parse_fn(example_serialized, is_train):
    name, image_buffer, label = parse_example_proto(example_serialized)
    
    image = preprocess_image(image_buffer=image_buffer, is_training=is_train)
    
    return name, image, label


def readTFRecord():
    """
    read TFRecord data (images).
    Arguments:
        tfName -- the TFRecord file to be read.
    Return: data saved in recordName (image and label).
    """
    #filenameQueue = tf.train.string_input_producer([tfName])
    dataset_fn = tf.data.TFRecordDataset
    _parse_fn = lambda x: parse_fn(x, is_train=False)
    
    # create a tf.data.Dataset from list of files
    filenames = tf.data.Dataset.list_files("./processed/train*.tfrecords")
    dataset = filenames.apply(
      tf.data.experimental.parallel_interleave(dataset_fn, cycle_length=4))
    
    dataset = dataset.map(_parse_fn, num_parallel_calls=2)
    print('After dataset map: {}'.format(dataset))
    
    return make_iterator(dataset)
    

def make_iterator(dataset):
    # only repeat dataset one time
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1024, count=1))
    dataset = dataset.batch(batch_size=2)
    dataset = dataset.prefetch(buffer_size=8)
    iterator = dataset.make_one_shot_iterator()

    return iterator
    

#preprocess('./data/', './processed/')
#createTFRecord('./processed/')

iterator = readTFRecord()
one_element = iterator.get_next()

with tf.Session() as sess:
    try:
        while True:
            #print(sess.run(one_element))
            name, img, lab = sess.run(one_element)
            print('name: {}, image shape: {}, label: {}'.format(name, img.shape, lab))
    except tf.errors.OutOfRangeError:
        print("End of iterator")
