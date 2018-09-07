# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com


#from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np

def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int64])

class LoadData(object):

    def __init__(self, data_path, filenames_file, num_threads, batch_size, height, width, mode):
        self.data_path = data_path
        self.num_threads = num_threads
        self.batch_size = batch_size

        self.height = height
        self.width = width

        self.mode = mode

        self.image_batch = None
        #self.left_image_batch  = None
        #self.right_image_batch = None

        with tf.name_scope('data_loader'):
            #files = tf.train.match_filenames_once(self.data_path+'*.jpg')
            input_queue = tf.train.string_input_producer([filenames_file], shuffle=False)

            line_reader = tf.TextLineReader()
            _, line = line_reader.read(input_queue)
            #split_line = tf.string_split([line]).values

            image_path  = tf.string_join([self.data_path, line])
            image_o  = self.read_image(image_path)
            #print '\t \t \t \t aaaaaaa'

            if mode == 'train':
                # randomly augment images
                #do_augment  = tf.random_uniform([], 0, 1)
                #image = tf.cond(do_augment > 0.5, lambda: self.augment_image(image_o), lambda: (image_o))
                #print image_new.shape
                #image = tf.reshape(image_new, [1, 128, 128, 1])

                self.image_batch = tf.stack(image_o,  0)
                self.image_batch = tf.reshape(self.image_batch, [1, 128, 128, 1])
                #self.image_batch = tf.reshape(self.image_batch, [1, 128, 128, 3])

            elif mode == 'test':
                #self.image_batch = image_o
                self.image_batch = tf.reshape(image_o, [1, 128, 128, 1])
                #self.image_batch = tf.reshape(image_o, [1, 128, 128, 3])
                self.im = image_o

    def augment_image(self, image):
        with tf.name_scope('data_augmentation'):
            # randomly shift gamma
            random_gamma = tf.random_uniform([], 0.8, 1.2)
            image_aug  = image  ** random_gamma

            # randomly shift brightness
            random_brightness = tf.random_uniform([], 0.5, 2.0)
            image_aug  =  image_aug * random_brightness

            # randomly shift color
            random_colors = tf.random_uniform([1], 0.8, 1.2)
            white = tf.ones([tf.shape(image)[0], tf.shape(image)[1]])
            color_image = tf.stack([white * random_colors[0]], axis=2)
            image_aug  *= color_image

            # saturate
            image_aug  = tf.clip_by_value(image_aug,  0, 1)

            return image_aug

    def read_image(self, image_path):
        with tf.name_scope('data_reader'):
            # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
            path_length = string_length_tf(image_path)[0]
            file_extension = tf.substr(image_path, path_length - 3, 3)
            file_cond = tf.equal(file_extension, 'jpg')

            # For gray image
            image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))
            image  = tf.image.rgb_to_grayscale(image)

            #image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path), channels=3), lambda: tf.image.decode_png(tf.read_file(image_path), channels=3))
            image  = tf.image.convert_image_dtype(image,  tf.float32)


            image = tf.divide(image, 255.)
            image  = tf.image.resize_images(image,  [128, 128], tf.image.ResizeMethod.AREA)

            print 'dataLoader\n',image,'\n'

            return image
