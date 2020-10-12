import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, confusion_matrix
from src.models.cnn_counts_student import CNN
import seaborn as sns
import tensorflow_probability as tfp
import csv
from tensorflow.contrib.tensorboard.plugins import projector

_EPSILON = 0.0001


class GaussObj:
    def __init__(self, mean, log_var, nb_latent, batch_size):
        super(GaussObj, self).__init__()
        self._nb_latent = nb_latent
        self._batch_size = batch_size
        self._log_var = log_var
        self._mean = tf.reshape(mean, [self._batch_size, -1, 1])
        self._std = tf.reshape(tf.sqrt(tf.exp(log_var)), [self._batch_size, -1, 1])
        self._cov = self._cal_cov_matrix(self._std)
        self._dist_epsilon = self._get_dist(tf.zeros_like(self._mean), tf.ones_like(self._mean))

    def _cal_cov_matrix(self, std):
        cov = tf.matrix_diag(tf.reshape(std, [self._batch_size, self._nb_latent]))
        return cov

    def sampler(self):
        with tf.variable_scope("sampler"):
            epsilon_sample = self._dist_epsilon.sample()
            z_sample = self._mean + tf.matmul(self._cov, tf.reshape(epsilon_sample, [self._batch_size, self._nb_latent, 1]))
        return z_sample

    def _get_dist(self, mean=None, std=None):
        mean = self._mean if mean is None else mean
        std = self._std if std is None else std
        mean = tf.reshape(mean, [self._batch_size, -1])
        std = tf.reshape(std, [self._batch_size, -1])
        prob = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=std)
        return prob

    def get_dist(self):
        return self._get_dist()

    def get_mean(self):
        return self._mean

    def get_log_var(self):
        return self._log_var


class ACC:
    def __init__(self, config, reuse=False):
        super(ACC, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config["callback"]['mode']
        self.nb_class = self.config["train"]["nb_class"]
        self.save_np = expt_config['save_np']
        self.mode = expt_config['mode']
        self.reuse = reuse
        self.filename = 'roc_' + str(self.mode) + '_' + str(self.mode)

    def inference(self, input, is_training, use_dr, mode):
        # setup the model, loss and metrics
        beta = tf.placeholder_with_default(1.0, shape=(), name='beta')
        loss_weight = tf.placeholder_with_default(1.0, shape=(), name='loss_weight')
        cnn = CNN(self.config['train'], is_training, use_dr)

        pred = cnn.build_model(input,
                               loss_weight,
                               beta,
                               self.config["callback"]['batch_size'],
                               checkpoint=None,
                               mode=mode,
                               reuse=self.reuse)

        return pred

    def on_epoch_end(self, callback_data, nb_imgs, mode_value, epoch):

        self.callback_data = callback_data
        nb_class = self.nb_class

        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        is_training = tf.placeholder(tf.bool)
        use_dr = tf.placeholder(tf.bool)
        mode = tf.placeholder(tf.bool)
        pred = self.inference(next_batch, is_training, use_dr, mode)

        y_true_labels = np.empty(nb_imgs)
        y_pred_labels = np.empty(nb_imgs)

        image_class = np.zeros((nb_imgs, nb_class))
        nb_images_class = np.zeros(nb_class)

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                batch, pred_label = sess.run([next_batch, pred], feed_dict={is_training: False,
                                                                            use_dr: False,
                                                                            mode: mode_value})

                y_true_label = batch[2][0]
                y_pred_label = np.argmax(pred_label[0])

                #print(y_true_label, pred_label, y_pred_label)

                y_true_labels[i] = y_true_label
                y_pred_labels[i] = y_pred_label

        for class_indx in range(nb_class):
            image_class[:, class_indx] = ((y_true_labels == class_indx) & (y_pred_labels == class_indx)).astype(float)
            nb_images_class[class_indx] = np.sum(y_true_labels == class_indx)

        acc = np.sum(image_class, axis=0) / (nb_images_class + _EPSILON)
        return acc


class UNC:
    def __init__(self, config, reuse=False, name='post'):
        super(UNC, self).__init__()
        self.callback_data = None
        self.config = config
        self.reuse = reuse
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config["callback"]['mode']
        self.save_np = expt_config['save_np']
        self.nb_latent = self.config["train"]["nb_latent"]
        self.nb_class = self.config["train"]["nb_class"]
        self.use_activity_loss = self.config["train"]["activity_loss"]
        self.filename = 'count_prob_' + str(self.mode) + '_' + name

    def inference(self, input_1, input_2, input_3, input_4, input_5, is_training, use_dr, mode):

        model = CNN(self.config['train'], is_training, use_dr)
        encoder = model.get_encoder()
        classifier = model.get_classifier()

        batch_size = self.config["callback"]['batch_size']
        reuse = self.reuse
        nb_latent = self.nb_latent

        z_post_mean, z_post_log_var = encoder(input_1, input_2, input_3, batch_size, reuse=reuse, name='teacher')
        dist_post = GaussObj(z_post_mean, z_post_log_var, nb_latent, batch_size)
        z_post_sample = dist_post.sampler()

        y_post_label_logit, y_post_label = classifier(z_post_sample,
                                                      batch_size=batch_size,
                                                      checkpoint=None,
                                                      reuse=reuse,
                                                      name='class-teacher')

        z_prior_mean, z_prior_log_var = encoder(input_1, input_4, input_5, batch_size, reuse=reuse, name='student')
        dist_prior = GaussObj(z_prior_mean, z_prior_log_var, nb_latent, batch_size)
        z_prior_sample = dist_prior.sampler()

        y_prior_label_logit, y_prior_label = classifier(z_prior_sample,
                                                        batch_size=batch_size,
                                                        checkpoint=None,
                                                        reuse=reuse,
                                                        name='class-student')
        if self.use_activity_loss:
            y_prior_label, y_pred_activity = y_prior_label
            y_post_label, _ = y_post_label

        y_label = tf.where(mode, y_post_label, y_prior_label)

        return y_label

    def on_epoch_end(self, callback_data, nb_imgs, nb_samples, epoch):

        self.callback_data = callback_data
        nb_class = self.nb_class

        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        is_training = tf.placeholder(tf.bool, name='is_training')
        use_dr = tf.placeholder(tf.bool, name='use_dr')
        mode = tf.placeholder(tf.bool, name='mode')

        input_1 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_2 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_3 = tf.placeholder(tf.float32, [1, 192, 192, 64])
        input_4 = tf.placeholder(tf.float32, [1, 192, 192, 64])
        input_5 = tf.placeholder(tf.float32, [1, 192, 192, 64])

        y_pred = self.inference(input_1, input_2, input_3, input_4, input_5, is_training, use_dr, mode)

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        entropy_correct = 0
        entropy_wrong = 0
        nb_correct = 0
        nb_wrong = 0

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                batch_np = sess.run(next_batch)

                y_true_label = batch_np[2][0]

                input_1_np = batch_np[0][..., :4]
                input_2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])
                input_3_np = batch_np[1][..., -1]

                input_4_np = batch_np[0][..., 4]
                input_5_np = batch_np[0][..., 5]

                y_pred_all = np.empty(nb_samples)

                for itr in range(nb_samples):
                    y_pred_label = sess.run(y_pred, feed_dict={is_training: False,
                                                               use_dr: False,
                                                               mode: False,
                                                               input_1: input_1_np,
                                                               input_2: input_2_np,
                                                               input_3: input_3_np,
                                                               input_4: input_4_np,
                                                               input_5: input_5_np
                                                               })

                    y_pred_all[itr] = np.argmax(y_pred_label[0])

                bins = np.arange(-0.5, nb_class + 0.5)
                n, _, _ = plt.hist(y_pred_all, bins, facecolor='green', edgecolor='black', alpha=0.75)
                p = n/nb_samples
                y_pred_label = np.argmax(p)

                if y_pred_label == y_true_label:
                    entropy_correct += -np.sum(np.log(p + _EPSILON) * p)
                    nb_correct += 1
                else:
                    entropy_wrong += -np.sum(np.log(p + _EPSILON) * p)
                    nb_wrong += 1

        entropy_correct = entropy_correct / (nb_correct + _EPSILON)
        entropy_wrong = entropy_wrong / (nb_wrong + _EPSILON)
        entropy = [entropy_correct, entropy_wrong]
        return entropy


class DISC:
    def __init__(self, config, reuse=False, name='post'):
        super(DISC, self).__init__()
        self.callback_data = None
        self.config = config
        self.reuse = reuse
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config["callback"]['mode']
        self.save_np = expt_config['save_np']
        self.prob = expt_config['prob']
        self.nb_latent = self.config["train"]["nb_latent"]
        self.nb_class = self.config["train"]["nb_class"]
        self.use_activity_loss = self.config["train"]["activity_loss"]
        self.filename = 'count_prob_' + str(self.mode) + '_' + name

    def inference(self, input_1, input_2, input_3, input_4, input_5, is_training, use_dr, mode):

        model = CNN(self.config['train'], is_training, use_dr)
        encoder = model.get_encoder()
        discriminator = model.get_discriminator()

        batch_size = self.config["callback"]['batch_size']
        reuse = self.reuse
        nb_latent = self.nb_latent

        if self.prob:
            z_post_mean, z_post_log_var = encoder(input_1, input_2, input_3, batch_size, reuse=reuse, name='teacher')
            dist_post = GaussObj(z_post_mean, z_post_log_var, nb_latent, batch_size)
            z_post_sample = dist_post.sampler()

            z_prior_mean, z_prior_log_var = encoder(input_1, input_4, input_5, batch_size, reuse=reuse, name='student')
            dist_prior = GaussObj(z_prior_mean, z_prior_log_var, nb_latent, batch_size)
            z_prior_sample = dist_prior.sampler()

            _, prob_fake = discriminator(z_prior_sample, batch_size, reuse=reuse)
            _, prob_real = discriminator(z_post_sample, batch_size, reuse=True)

        else:
            z_post_mean = encoder(input_1, input_2, input_3, batch_size, reuse=reuse, name='teacher')
            z_prior_mean = encoder(input_1, input_4, input_5, batch_size, reuse=reuse, name='student')

            _, prob_fake = discriminator(z_prior_mean, batch_size, reuse=reuse)
            _, prob_real = discriminator(z_post_mean, batch_size, reuse=True)

        return prob_fake, prob_real

    def on_epoch_end(self, callback_data, nb_imgs, epoch):
        self.callback_data = callback_data

        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        is_training = tf.placeholder(tf.bool, name='is_training')
        use_dr = tf.placeholder(tf.bool, name='use_dr')
        mode = tf.placeholder(tf.bool)

        input_1 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_2 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_3 = tf.placeholder(tf.float32, [1, 192, 192, 64])
        input_4 = tf.placeholder(tf.float32, [1, 192, 192, 64])
        input_5 = tf.placeholder(tf.float32, [1, 192, 192, 64])

        z_post_mean_tf, z_prior_mean_tf = self.inference(input_1, input_2, input_3, input_4, input_5, is_training, use_dr, mode)

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            prob_fake_all = 0.
            prob_real_all = 0.

            for i in range(nb_imgs):

                batch_np = sess.run(next_batch)

                input_1_np = batch_np[0][..., :4]
                input_2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])
                input_3_np = batch_np[1][..., -1] #+ np.random.normal(loc=0.0, scale=.1, size=[192, 192, 64])
                input_4_np = batch_np[0][..., 4]
                input_5_np = batch_np[0][..., 5]

                prob_fake, prob_real = sess.run([z_post_mean_tf, z_prior_mean_tf],
                                                feed_dict={is_training: False,
                                                           use_dr: False,
                                                           input_1: input_1_np,
                                                           input_2: input_2_np,
                                                           input_3: input_3_np,
                                                           input_4: input_4_np,
                                                           input_5: input_5_np})

                prob_fake_all += prob_fake[0][0]
                prob_real_all += prob_real[0][0]

        prob_fake_all = prob_fake_all/nb_imgs
        prob_real_all = prob_real_all / nb_imgs

        return prob_real_all, prob_fake_all


class Dist:
    def __init__(self, config, reuse=False, name='post'):
        super(Dist, self).__init__()
        self.callback_data = None
        self.config = config
        self.reuse = reuse
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config["callback"]['mode']
        self.save_np = expt_config['save_np']
        self.nb_latent = self.config["train"]["nb_latent"]
        self.nb_class = self.config["train"]["nb_class"]
        self.use_activity_loss = self.config["train"]["activity_loss"]
        self.filename = 'count_prob_' + str(self.mode) + '_' + name

    def inference(self, input_1, input_2, input_3, input_4, input_5, is_training, use_dr, mode):

        model = CNN(self.config['train'], is_training, use_dr)
        encoder = model.get_encoder()
        classifier = model.get_classifier()

        batch_size = self.config["callback"]['batch_size']
        reuse = self.reuse
        nb_latent = self.nb_latent

        z_post_mean, z_post_log_var = encoder(input_1, input_2, input_3, batch_size, reuse=reuse, name='teacher')
        dist_post = GaussObj(z_post_mean, z_post_log_var, nb_latent, batch_size)
        z_post_sample = dist_post.sampler()

        y_post_label_logit, y_post_label = classifier(z_post_sample,
                                                      batch_size=batch_size,
                                                      checkpoint=None,
                                                      reuse=reuse,
                                                      name='class-teacher')

        z_prior_mean, z_prior_log_var = encoder(input_1, input_4, input_5, batch_size, reuse=reuse, name='student')
        dist_prior = GaussObj(z_prior_mean, z_prior_log_var, nb_latent, batch_size)
        z_prior_sample = dist_prior.sampler()

        y_prior_label_logit, y_prior_label = classifier(z_prior_sample,
                                                        batch_size=batch_size,
                                                        checkpoint=None,
                                                        reuse=reuse,
                                                        name='class-student')
        if self.use_activity_loss:
            y_prior_label, y_pred_activity = y_prior_label
            y_post_label, _ = y_post_label

        dist_z = tf.norm(z_post_mean - z_prior_mean)
        dist_y = tf.norm(tf.cast(tf.math.argmax(y_post_label, axis=1) - tf.math.argmax(y_prior_label, axis=1), tf.float32))

        return dist_z, dist_y

    def on_epoch_end(self, callback_data, nb_imgs, epoch):

        self.callback_data = callback_data

        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        is_training = tf.placeholder(tf.bool, name='is_training')
        use_dr = tf.placeholder(tf.bool, name='use_dr')
        mode = tf.placeholder(tf.bool)

        input_1 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_2 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_3 = tf.placeholder(tf.float32, [1, 192, 192, 64])
        input_4 = tf.placeholder(tf.float32, [1, 192, 192, 64])
        input_5 = tf.placeholder(tf.float32, [1, 192, 192, 64])

        dist_z_tf, dist_y_tf = self.inference(input_1, input_2, input_3, input_4, input_5, is_training, use_dr, mode)

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            dist_z_all = np.empty(nb_imgs)
            dist_y_all = np.empty(nb_imgs)

            for i in range(nb_imgs):

                batch_np = sess.run(next_batch)

                input_1_np = batch_np[0][..., :4]
                input_2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])
                input_3_np = batch_np[1][..., -1]
                input_4_np = batch_np[0][..., 4]
                input_5_np = batch_np[0][..., 5]

                dist_z, dist_y = sess.run([dist_z_tf, dist_y_tf],
                                          feed_dict={is_training: False,
                                                     use_dr: False,
                                                     input_1: input_1_np,
                                                     input_2: input_2_np,
                                                     input_3: input_3_np,
                                                     input_4: input_4_np,
                                                     input_5: input_5_np})
                dist_y_all[i] = dist_y
                dist_z_all[i] = dist_z
        dist_y_avg = np.mean(dist_y_all)
        dist_z_avg = np.mean(dist_z_all)
        return dist_z_avg, dist_y_avg


class DistZ:
    def __init__(self, config, reuse=False, name='post'):
        super(DistZ, self).__init__()
        self.callback_data = None
        self.config = config
        self.reuse = reuse
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config["callback"]['mode']
        self.save_np = expt_config['save_np']
        self.nb_latent = self.config["train"]["nb_latent"]
        self.nb_class = self.config["train"]["nb_class"]
        self.use_activity_loss = self.config["train"]["activity_loss"]
        self.filename = 'count_prob_' + str(self.mode) + '_' + name

    def inference(self, input_1, input_2, input_3, input_4, input_5, is_training, use_dr, mode):

        model = CNN(self.config['train'], is_training, use_dr)
        encoder = model.get_encoder()

        batch_size = self.config["callback"]['batch_size']
        reuse = self.reuse

        z_post_mean, z_post_log_var = encoder(input_1, input_2, input_3, batch_size, reuse=reuse, name='teacher')
        z_prior_mean, z_prior_log_var = encoder(input_1, input_4, input_5, batch_size, reuse=reuse, name='student')

        return z_post_mean, z_prior_mean

    def on_epoch_end(self, callback_data, nb_imgs, epoch):

        self.callback_data = callback_data

        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        is_training = tf.placeholder(tf.bool, name='is_training')
        use_dr = tf.placeholder(tf.bool, name='use_dr')
        mode = tf.placeholder(tf.bool)

        input_1 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_2 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_3 = tf.placeholder(tf.float32, [1, 192, 192, 64])
        input_4 = tf.placeholder(tf.float32, [1, 192, 192, 64])
        input_5 = tf.placeholder(tf.float32, [1, 192, 192, 64])

        z_post_mean_tf, z_prior_mean_tf = self.inference(input_1, input_2, input_3, input_4, input_5, is_training, use_dr, mode)

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            z_post_mean_all = np.zeros((nb_imgs, self.nb_latent))
            z_prior_mean_all = np.zeros((nb_imgs, self.nb_latent))
            y_true_all = np.empty(nb_imgs)
            diff_z_post_class = np.zeros((self.nb_class, self.nb_class))
            diff_z_prior_class = np.zeros((self.nb_class, self.nb_class))

            for i in range(nb_imgs):

                batch_np = sess.run(next_batch)

                input_1_np = batch_np[0][..., :4]
                input_2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])
                input_3_np = batch_np[1][..., -1]
                input_4_np = batch_np[0][..., 4]
                input_5_np = batch_np[0][..., 5]

                y_true = batch_np[2][0]

                z_post_mean, z_prior_mean = sess.run([z_post_mean_tf, z_prior_mean_tf],
                                                     feed_dict={is_training: False,
                                                                use_dr: False,
                                                                input_1: input_1_np,
                                                                input_2: input_2_np,
                                                                input_3: input_3_np,
                                                                input_4: input_4_np,
                                                                input_5: input_5_np})

                z_post_mean = np.reshape(z_post_mean, [1, self.nb_latent])[0]
                z_prior_mean = np.reshape(z_prior_mean, [1, self.nb_latent])[0]

                z_post_mean_all[i] = z_post_mean
                z_prior_mean_all[i] = z_prior_mean
                y_true_all[i] = y_true

            for i in range(self.nb_class):
                z_post_mean_class_i = z_post_mean_all[y_true_all == i]
                z_prior_mean_class_i = z_prior_mean_all[y_true_all == i]
                for j in range(i, self.nb_class):
                    z_post_mean_class_j = z_post_mean_all[y_true_all == j]
                    z_prior_mean_class_j = z_prior_mean_all[y_true_all == j]
                    nb_samples_class_i = len(z_post_mean_class_i)
                    nb_samples_class_j = len(z_post_mean_class_j)
                    if (nb_samples_class_i==0) or (nb_samples_class_j==0):
                        diff_z_post_class[i, j] = 0
                        diff_z_post_class[j, i] = 0
                        diff_z_prior_class[i, j] = 0
                        diff_z_prior_class[j, i] = 0
                    else:
                        z_post_mean_class_i_rep = np.repeat(z_post_mean_class_i, nb_samples_class_j, axis=0)
                        z_post_mean_class_j_rep = np.tile(z_post_mean_class_j, [nb_samples_class_i, 1])
                        z_prior_mean_class_i_rep = np.repeat(z_prior_mean_class_i, nb_samples_class_j, axis=0)
                        z_prior_mean_class_j_rep = np.tile(z_prior_mean_class_j, [nb_samples_class_i, 1])
                        z_post_dist = np.mean(np.sqrt(np.sum((z_post_mean_class_i_rep - z_post_mean_class_j_rep) ** 2, axis=1)))
                        z_prior_dist = np.mean(np.sqrt(np.sum((z_prior_mean_class_i_rep - z_prior_mean_class_j_rep) ** 2, axis=1)))
                        diff_z_post_class[i, j] = z_post_dist
                        diff_z_post_class[j, i] = z_post_dist
                        diff_z_prior_class[i, j] = z_prior_dist
                        diff_z_prior_class[j, i] = z_prior_dist

            between_class_z_post = np.sum(np.tril(diff_z_post_class, -1))/self.nb_class
            within_class_z_post = np.sum(np.diag(diff_z_post_class))/self.nb_class
            between_class_z_prior = np.sum(np.tril(diff_z_prior_class, -1))/self.nb_class
            within_class_z_prior = np.sum(np.diag(diff_z_prior_class))/self.nb_class
        return between_class_z_post, within_class_z_post, between_class_z_prior, within_class_z_prior


class PlotCM:
    def __init__(self, config, reuse=False, name='prior'):
        super(PlotCM, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.save_np = expt_config['save_np']
        self.reuse = reuse
        self.plot_softman = False
        self.filename = 'cm_' + str(self.mode) + '_' + str(name)

    def inference(self, input, is_training, use_dr, mode, checkpoint=None):
        # setup the model, loss and metrics
        beta = tf.placeholder_with_default(1.0, shape=(), name='beta')
        loss_weight = tf.placeholder_with_default(1.0, shape=(), name='loss_weight')
        cnn = CNN(self.config['train'], is_training, use_dr)

        pred = cnn.build_model(input,
                               loss_weight,
                               beta,
                               self.config['callback']['batch_size'],
                               mode,
                               checkpoint,
                               self.reuse)
        return pred


    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, fig,
                              normalize=True):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(cm)
        ax = fig.add_subplot(1, 1, 1)
        print(cm)
        g1 = sns.heatmap(cm, cmap=plt.cm.Blues, ax=ax, cbar=False, annot=True)
        g1.set_ylabel('')
        g1.set_xlabel('')
        bottom, top = ax.get_ylim()

        plt.setp(ax.get_yticklabels(), rotation=90,
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        ax.set_ylim(bottom + 1, top - 1)
        return ax

    def on_epoch_end(self, callback_data, nb_imgs, epoch):

        self.callback_data = callback_data
        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        is_training = tf.placeholder(tf.bool)
        use_dr = tf.placeholder(tf.bool)
        mode = tf.placeholder(tf.bool)

        #checkpoint = os.path.join("/cim/nazsepah/projects/counts3d-tf/results/miccai/cnn-count-prob-gauss-multi-68/checkpoints/model.ckpt-0")
        pred = self.inference(next_batch, is_training, use_dr, mode, checkpoint=None)
        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        y_true_labels = np.empty(nb_imgs)
        y_pred_labels = np.empty(nb_imgs)

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                batch, pred_label = sess.run([next_batch, pred], feed_dict={is_training: False,
                                                                            use_dr: False,
                                                                            mode: False})

                y_true_label = batch[2][0]
                y_pred_label = np.argmax(pred_label[0])

                if self.plot_softman:
                    if i < 30:
                        fig = plt.figure()
                        plt.bar(np.arange(len(pred_label[0])), pred_label[0])
                        plt.title("gt count:{}".format(y_true_label))
                        fig.savefig(
                            os.path.join(self.outdir, self.expt_name, "recons", self.filename + "_softmax_output_{}_{}.png".format(i, epoch)))
                        plt.close()

                y_true_labels[i] = y_true_label
                y_pred_labels[i] = y_pred_label

            fig = plt.figure()
            PlotCM.plot_confusion_matrix(y_true_labels, y_pred_labels, fig=fig)
            fig.savefig(os.path.join(self.outdir, self.expt_name, "recons", self.filename +"_{}.png".format(epoch)))
            plt.close()


class PlotTSNE:
    def __init__(self, config, callback_data, reuse):
        super(PlotTSNE, self).__init__()
        self.callback_data = callback_data
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.nb_latent = self.config["train"]["nb_latent"]
        self.reuse = reuse
        self.filename = 'tsne_' + str(self.mode)

    def inference(self, input_1, input_2, input_3, is_training, use_dr):

        model = CNN(self.config['train'], is_training, use_dr)
        encoder = model.get_encoder()

        batch_size = self.config['callback']['batch_size']
        reuse = self.reuse
        nb_latent = self.nb_latent

        z_post_mean, z_post_log_var = encoder(input_1, input_2, input_3, batch_size, reuse=reuse, name='teacher')
        dist_post = GaussObj(z_post_mean, z_post_log_var, nb_latent, batch_size)
        z_post_sample = dist_post.sampler()

        z_prior_mean, z_prior_log_var = encoder(input_1, input_2, None, batch_size, reuse=reuse, name='student')
        dist_prior = GaussObj(z_prior_mean, z_prior_log_var, nb_latent, batch_size)
        z_prior_sample = dist_prior.sampler()

        return z_post_sample, z_post_mean, z_prior_sample, z_prior_mean

    def on_epoch_end(self, nb_imgs, epoch):

        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        is_training = tf.placeholder(tf.bool, name='is_training')
        use_dr = tf.placeholder(tf.bool, name='use_dr')

        nb_latent = self.nb_latent

        input_1 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_2 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_3 = tf.placeholder(tf.float32, [1, 192, 192, 64])

        z_sample_post, z_mean_post, z_sample_prior, z_mean_prior = self.inference(input_1, input_2, input_3, is_training, use_dr)

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        folder_name = 'log-post-'

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            means = np.zeros((nb_imgs, nb_latent))
            labels = [[[''] for i in range(3)] for j in range(nb_imgs + 1)]

            labels[0][0] = 'post/prior'
            labels[0][1] = 'img_indx'
            labels[0][2] = 'count'

            os.makedirs(os.path.join(self.outdir, self.expt_name, folder_name + self.mode, "log-mean-{}".format(epoch)), exist_ok=True)

            for i in range(nb_imgs):

                print(i)

                batch_np = sess.run(next_batch)

                input_1_np = batch_np[0][..., :4]
                input_2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])
                input_3_np = batch_np[1][..., -1] + np.random.normal(loc=0.0, scale=.1, size=[192, 192, 64])
                y_true = batch_np[2][0]

                z_post, z_mu_post, z_prior, z_mu_prior = sess.run([z_sample_post,
                                                                   z_mean_post,
                                                                   z_sample_prior,
                                                                   z_mean_prior],
                                                                  feed_dict={is_training: False,
                                                                             use_dr: False,
                                                                             input_1: input_1_np,
                                                                             input_2: input_2_np,
                                                                             input_3: input_3_np})

                labels[i+1][0] = '1'
                labels[i+1][1] = str(i)
                labels[i+1][2] = str(y_true)
                means[i, :] = z_mu_post[0].flatten()

            with open(os.path.join(self.outdir, self.expt_name, folder_name + self.mode, "log-mean-{}".format(epoch), 'metadata.tsv'), "w", newline ='') as tsvfile:
                writer = csv.writer(tsvfile, delimiter="\t")
                writer.writerows(labels)

            tf_data = tf.Variable(means, name='tf_means')

            saver = tf.train.Saver([tf_data])
            sess.run(tf_data.initializer)
            saver.save(sess, os.path.join(self.outdir, self.expt_name, folder_name + self.mode, "log-mean-{}".format(epoch), 'tf_data.ckpt'))

            config = projector.ProjectorConfig()  # One can add multiple embeddings.
            embedding = config.embeddings.add()
            embedding.tensor_name = tf_data.name  # Link this tensor to its metadata(Labels) file
            embedding.metadata_path = os.path.join(self.outdir, self.expt_name, folder_name + self.mode, "log-mean-{}".format(epoch), 'metadata.tsv')  # Saves a config file that TensorBoard will read during startup.
            projector.visualize_embeddings(
                tf.summary.FileWriter(os.path.join(self.outdir, self.expt_name, folder_name + self.mode, "log-mean-{}".format(epoch))), config)


class PlotCMProb:
    def __init__(self, config, callback_data, reuse=False, name='post'):
        super(PlotCMProb, self).__init__()
        self.callback_data = callback_data
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.nb_latent = self.config["train"]["nb_latent"]
        self.nb_class = self.config["train"]["nb_class"]
        self.reuse = reuse
        self.filename = 'cm_prob_' + str(self.mode) + '_' + str(name)

    def inference(self, input_1, input_2, input_3, is_training, use_dr, mode):

        model = CNN(self.config['train'], is_training, use_dr)
        encoder = model.get_encoder()
        classifier = model.get_classifier()

        batch_size = self.config['callback']['batch_size']
        reuse = self.reuse
        nb_latent = self.nb_latent

        z_post_mean, z_post_log_var = encoder(input_1, input_2, input_3, batch_size, reuse=reuse, name='teacher')
        dist_post = GaussObj(z_post_mean, z_post_log_var, nb_latent, batch_size)
        z_post_sample = dist_post.sampler()

        y_post_label_logit, y_post_label = classifier(z_post_sample,
                                                      batch_size=batch_size,
                                                      checkpoint=None,
                                                      reuse=reuse,
                                                      name='class-teacher')

        z_prior_mean, z_prior_log_var = encoder(input_1, input_2, None, batch_size, reuse=reuse, name='student')
        dist_prior = GaussObj(z_prior_mean, z_prior_log_var, nb_latent, batch_size)
        z_prior_sample = dist_prior.sampler()

        y_prior_label_logit, y_prior_label = classifier(z_prior_sample,
                                                        batch_size=batch_size,
                                                        checkpoint=None,
                                                        reuse=reuse,
                                                        name='class-student')

        y_prior_label, y_pred_activity = y_prior_label
        y_post_label, _ = y_post_label

        y_label = tf.where(mode, y_post_label, y_prior_label)

        return y_label


    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, fig,
                              normalize=False):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        # print(cm)
        ax = fig.add_subplot(1, 1, 1)
        print(cm)
        g1 = sns.heatmap(cm, cmap="YlGnBu", ax=ax, cbar=False, annot=True)
        g1.set_ylabel('')
        g1.set_xlabel('')
        bottom, top = ax.get_ylim()

        plt.setp(ax.get_yticklabels(), rotation=90, rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        ax.set_ylim(bottom + 1, top - 1)
        return ax

    def on_epoch_end(self, nb_imgs, nb_samples, epoch):

        nb_class = self.nb_class

        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        is_training = tf.placeholder(tf.bool, name='is_training')
        use_dr = tf.placeholder(tf.bool, name='use_dr')
        mode = tf.placeholder(tf.bool)

        input_1 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_2 = tf.placeholder(tf.float32, [1, 192, 192, 64, 4])
        input_3 = tf.placeholder(tf.float32, [1, 192, 192, 64])

        y_pred = self.inference(input_1, input_2, input_3, is_training, use_dr, mode)

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        y_true_labels = []
        y_pred_labels1 = np.empty(nb_imgs)
        y_pred_labels2 = np.empty(nb_imgs)
        y_pred_labels3 = []
        y_true_labels_all = np.empty(nb_imgs)
        dis_ent = np.empty(nb_imgs)
        ent_thre = .25

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                print(i)

                batch_np = sess.run(next_batch)

                input_1_np = batch_np[0][..., :4]
                input_2_np = (batch_np[1][..., :4] - batch_np[0][..., :4])
                input_3_np = batch_np[1][..., -1] + np.random.normal(loc=0.0, scale=.1, size=[192, 192, 64])

                y_pred_all = np.empty((nb_samples, nb_class))

                y_true_labels_all[i] = batch_np[2][0]

                for itr in range(nb_samples):

                    y_pred_label = sess.run(y_pred, feed_dict={is_training: False,
                                                               use_dr: False,
                                                               input_1: input_1_np,
                                                               input_2: input_2_np,
                                                               input_3: input_3_np,
                                                               mode: True
                                                               })

                    y_pred_all[itr, :] = y_pred_label[0]
                    y_pred_labels1[i] = np.argmax(y_pred_label[0])

                # if len(np.unique(y_pred_all)) > 1:
                out = np.argmax(y_pred_all, axis=1)
                prob = np.zeros(nb_class)
                for j in range(nb_class):
                    prob[j] = np.sum(out == j) / nb_samples
                dis_ent[i] = -np.sum(prob * np.log2(prob + _EPSILON))

                y_pred_labels2[i] = np.argmax(prob)

                if dis_ent[i] < ent_thre:
                    y_true_labels.append(batch_np[2][0])
                    y_pred_labels3.append(np.argmax(prob))

            fig = plt.figure()
            PlotCM.plot_confusion_matrix(y_true_labels_all, y_pred_labels1, fig=fig)
            fig.savefig(
                os.path.join(self.outdir, self.expt_name, "recons", "CM_one_example_prior_{}_{}.png".format(ent_thre, epoch)))
            plt.close()

            fig = plt.figure()
            PlotCM.plot_confusion_matrix(y_true_labels_all, y_pred_labels2, fig=fig)
            fig.savefig(
                os.path.join(self.outdir, self.expt_name, "recons", "CM_avg_prior_{}_{}.png".format(ent_thre, epoch)))
            plt.close()

            fig = plt.figure()
            PlotCM.plot_confusion_matrix(y_true_labels, y_pred_labels3, fig=fig)
            fig.savefig(
                os.path.join(self.outdir, self.expt_name, "recons", "CM_unc_prior_{}_{}.png".format(ent_thre, epoch)))
            plt.close()


class PlotROC:
    def __init__(self, config, reuse=False, name='prior'):
        super(PlotROC, self).__init__()
        self.callback_data = None
        self.config = config
        expt_config = config['experiment']
        self.outdir = expt_config['outdir']
        self.expt_name = expt_config['name']
        self.mode = config['callback']['mode']
        self.save_np = expt_config['save_np']
        self.mode = expt_config['mode']
        self.reuse = reuse
        self.filename = 'roc_' + str(self.mode) + '_' + str(name)

    def inference(self, input, is_training, use_dr, mode, checkpoint=None):
        # setup the model, loss and metrics
        beta = tf.placeholder_with_default(1.0, shape=(), name='beta')
        loss_weight = tf.placeholder_with_default(1.0, shape=(), name='loss_weight')
        cnn = CNN(self.config['train'], is_training, use_dr)

        pred = cnn.build_model(input,
                               loss_weight,
                               beta,
                               self.config['callback']['batch_size'],
                               mode,
                               checkpoint,
                               self.reuse)
        return pred

    def on_epoch_end(self, callback_data, nb_imgs, epoch):

        self.callback_data = callback_data

        data_iterator = tf.data.Iterator.from_structure(self.callback_data.output_types)
        callback_init_op = data_iterator.make_initializer(self.callback_data)
        next_batch = data_iterator.get_next()

        is_training = tf.placeholder(tf.bool)
        use_dr = tf.placeholder(tf.bool)
        mode = tf.placeholder(tf.bool)
        pred = self.inference(next_batch, is_training, use_dr, mode)

        y_true_labels = np.empty(nb_imgs)
        y_pred_labels = np.empty(nb_imgs)

        checkpoint = os.path.join(self.outdir, self.expt_name, "checkpoints", "model.ckpt-{}".format(epoch))
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
            sess.run(callback_init_op)
            saver.restore(sess, checkpoint)

            for i in range(nb_imgs):

                batch, pred_label = sess.run([next_batch, pred], feed_dict={is_training: False,
                                                                            use_dr: False,
                                                                            mode: False})

                y_true_count = batch[2][0]
                y_true_label = int(y_true_count == 1)
                y_pred_label = pred_label[0][1]

                y_true_labels[i] = y_true_label
                y_pred_labels[i] = y_pred_label

            fpr_mean, tpr_mean, thresholds_c = roc_curve(y_true_labels, y_pred_labels, pos_label=1)

            if self.save_np:
                np.save(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "_tpr_{:03d}_1.npy".format(epoch)), tpr_mean)
                np.save(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "_fpr_{:03d}_1.npy".format(epoch)), fpr_mean)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.plot(fpr_mean, tpr_mean, label='fp-tp_scikit')
            plt.plot(fpr_mean, fpr_mean, label='random')
            plt.legend(loc="lower right")
            major_ticks = np.arange(0, 1, 0.1)
            minor_ticks = np.arange(0, 1, 0.02)
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(major_ticks)
            ax.set_yticks(minor_ticks, minor=True)
            ax.grid(which='both')
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.5)
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.set_title("auc:{}".format(auc(fpr_mean, tpr_mean)))
            fig.savefig(os.path.join(self.outdir, self.expt_name, "rocs", self.filename + "_{}_1.png".format(epoch)))
            plt.close()
