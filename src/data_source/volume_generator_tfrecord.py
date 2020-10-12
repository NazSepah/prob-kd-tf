import tensorflow as tf
import numpy as np
from os.path import join
import glob
import tensorflow_probability as tfp


_LOC = [0.1, 0.7, 1.9, 3.4, 3.8]
_SCALE = [0.1, 0.2, 0.4, 0.2, 0.1]
_SKEWNESS = [-0.1, 1.5, 0.5, -1.5, 2.0]


class BrainDataProvider:
    """
    Data provider for multimodal brain MRI.
    Usage:
    :param config: dict to config options
    :param outdir: path to the tfrecord files
    """

    def __init__(self, outdir, config):
        self._dim = config.get('img_shape')
        self._mode = config.get('mode')
        self._batch_size = config.get('batch_size', 1)
        self._augment = config.get("data_augment", False)
        self._rotate = config.get("data_rotate", False)
        self._nb_class = config.get("nb_class", 5)
        self._label_dist = config.get("label_dist", False)
        self._combined_labels = config.get("combined_labels", True)
        self._nb_samples = 1

        if self._mode == 'train':
            self._filenames_bravo = glob.glob(join(outdir, "bravo*train*.tfrecords"))
            self._filenames_ascend = glob.glob(join(outdir, "ascend_placebo*train*.tfrecords"))
            self._filename = self._filenames_bravo# + self._filenames_ascend
        elif self._mode == 'valid':
            self._filenames_bravo = glob.glob(join(outdir, 'bravo*valid*.tfrecords'))
            self._filenames_ascend = glob.glob(join(outdir, 'ascend_placebo*valid*.tfrecords'))
            self._filename = self._filenames_bravo# + self._filenames_ascend
        elif self._mode == 'test':
            self._filenames_bravo = glob.glob(join(outdir, 'bravo*test*.tfrecords'))
            self._filenames_ascend = glob.glob(join(outdir, 'ascend_placebo*test*.tfrecords'))
            self._filename = self._filenames_bravo

    def get_nb_samples(self):
        count = 0
        for fn in self._filename:
            for _ in tf.python_io.tf_record_iterator(fn):
                count += 1
        return count

    def combined_labels(self, nb_lesions):
        def f1(): return tf.constant(0)

        def f2(): return tf.constant(1)

        def f3(): return tf.constant(2)

        return tf.switch_case(nb_lesions, branch_fns={0: f1, 1: f2, 2: f2, 3: f2, 4: f2})

    def data_generator(self):

        def parser(serialized_example):
            """Parses a single tf.Example into image and label tensors."""
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'tp1': tf.FixedLenFeature([], tf.string),
                    'tp2': tf.FixedLenFeature([], tf.string),
                    'nb_newt2': tf.FixedLenFeature([], tf.int64),
                })

            input_images = tf.decode_raw(features['tp1'], tf.float32)
            input_images = tf.reshape(input_images, shape=(192, 192, 64, 6))
            output_images = tf.decode_raw(features['tp2'], tf.float32)
            output_images = tf.reshape(output_images, shape=(192, 192, 64, 5))
            nb_lesions_newt2 = tf.cast(features['nb_newt2'], tf.int32)

            if self._combined_labels:
                nb_lesions = self.combined_labels(nb_lesions_newt2)

            nb_lesions = tf.cast(tf.clip_by_value(nb_lesions, 0, self._nb_class - 1), tf.int32)
            label = tf.cast(tf.greater(nb_lesions_newt2, 0), tf.float32)
            return input_images, output_images, nb_lesions, label

        def label_dist(input_image, output_image, nb_lesions, label):

            nb_lesions = tf.cast(nb_lesions, tf.int32)
            count_one_hot = tf.one_hot(nb_lesions, self._nb_class)

            loc = tf.constant(np.expand_dims(np.array(_LOC), axis=1), tf.float32)
            scale = tf.constant(np.expand_dims(np.array(_SCALE), axis=1), tf.float32)
            skewness = tf.constant(np.expand_dims(np.array(_SKEWNESS), axis=1), tf.float32)
            tailweight = tf.constant(np.expand_dims(np.array([1., 1., 1., 1., 1]), axis=1), tf.float32)

            loc = tf.matmul(count_one_hot, loc)
            scale = tf.matmul(count_one_hot, scale)
            skewness = tf.matmul(count_one_hot, skewness)
            tailweight = tf.matmul(count_one_hot, tailweight)

            y_true_soft = tfp.distributions.SinhArcsinh(loc=loc, scale=scale, skewness=skewness, tailweight=tailweight).sample(sample_shape=self._nb_samples)
            y_true_soft = tf.cast(tf.reshape(y_true_soft, [self._batch_size, 1]), tf.float32)
            y_true_soft_rep = tf.tile(y_true_soft, [1, self._nb_class])
            y_true_soft_one_hot = count_one_hot * y_true_soft_rep

            return input_image, output_image, nb_lesions, y_true_soft_one_hot

        def augmentation(input_image, output_image, nb_lesions, label):

            random_angles = tf.random_uniform(shape=(), minval=-np.pi / 4, maxval=np.pi / 4)

            image = tf.concat([input_image, output_image], axis=-1)

            image = tf.reshape(image, [192, 192, 64 * 11])

            image_r = tf.contrib.image.rotate(image, random_angles)
            image_r = tf.image.random_flip_left_right(image_r)
            image_r = tf.image.random_flip_up_down(image_r)

            image_r = tf.reshape(image_r, [192, 192, 64, 11])

            input_image_r = image_r[..., :6]
            output_image_r = image_r[..., 6:]

            shift_value = tf.cast(tf.random_uniform((), -3, 3), tf.int32)
            input_image_r = tf.manip.roll(input_image_r, shift_value, axis=0)
            output_image_r = tf.manip.roll(output_image_r, shift_value, axis=0)

            shift_value = tf.cast(tf.random_uniform((), -12, 12), tf.int32)
            input_image_r = tf.manip.roll(input_image_r, shift_value, axis=1)
            output_image_r = tf.manip.roll(output_image_r, shift_value, axis=1)

            shift_value = tf.cast(tf.random_uniform((), -3, 3), tf.int32)
            input_image_r = tf.manip.roll(input_image_r, shift_value, axis=2)
            output_image_r = tf.manip.roll(output_image_r, shift_value, axis=2)

            return input_image_r, output_image_r, nb_lesions, label

        dataset = tf.data.TFRecordDataset(self._filename)
        if self._mode == 'train':
            dataset = dataset.shuffle(buffer_size=10 * self._batch_size)
        dataset = dataset.map(parser, num_parallel_calls=8)
        if (self._mode == 'train') & self._augment:
           dataset = dataset.map(augmentation, num_parallel_calls=8)
        dataset = dataset.repeat().batch(self._batch_size).prefetch(1)
        if self._label_dist:
            dataset = dataset.map(label_dist, num_parallel_calls=8)
        return dataset
