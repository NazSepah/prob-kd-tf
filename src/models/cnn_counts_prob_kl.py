import tensorflow as tf
from src.utils.ops import conv3d, conv_res_block, dense3d, conv_block
import tensorflow_probability as tfp


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


class CNN:
    """
    Variational Autoencoder
    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    def __init__(self, config, is_training, use_dr):
        super(CNN, self).__init__()
        self._is_training = is_training
        self._use_dr = use_dr
        self._image_shape = config.get("image_shape", [192, 192, 64, 4])
        self._base_num_filter = config.get("base_num_filter", 16)
        self._num_layers = config.get("num_layers", 5)
        self._dr_rate = config.get("dr_rate", 0.2)
        self._combined_labels = config.get("combined_labels", False)
        self._nb_class = config.get("nb_class", 5)
        self._weight_decay = config.get("weight_decay", False)
        self._use_disc_loss = config.get("disc_loss", False)
        self._use_soft_label = config.get("soft_label", False)
        self._use_grl_loss = config.get("grl_loss", False)
        self._use_activity_loss = config.get("activity_loss", False)
        self._use_kl_loss = config.get("kl_loss", False)
        self._lr = config.get("lr", 2e-4)
        self._beta1 = config.get("adam_beta1", 0.9)
        self._nb_latent = config.get("nb_latent", 72)
        self._broadcast = config.get("broadcast", True)
        self._soft_labels = config.get("soft_labels", False)
        self._add_noise = config.get("add_noise", False)
        self._temperature = 1.
        self._mode = None
        self._beta = None
        self._class_weight = None
        self._dist_post = None
        self._dist_prior = None
        self._class_weight = None
        self._y_true_label = None
        self._y_post_label_soft = None
        self._y_prior_label_soft_logit = None
        self._y_prior_label_true_logit = None
        self._prob_fake_logits = None
        self._prob_real_logits = None

    def build_model(self, batch, class_weight, beta, batch_size, mode, checkpoint, reuse=False):

        input_encoder_1 = batch[0][..., :4]
        input_encoder_2 = (batch[1][..., :4] - batch[0][..., :4])

        if self._add_noise:
            input_encoder_3 = batch[1][..., -1] + tf.random.normal([192, 192, 64], mean=0.0, stddev=.1)
        else:
            input_encoder_3 = batch[1][..., -1]

        input_encoder_4 = batch[0][..., 4]
        input_encoder_5 = batch[0][..., 5]

        self._y_true_label = batch[2]
        self._beta = beta
        self._class_weight = class_weight
        self._mode = mode

        ### get the parameters of posterior dist
        z_post_mean,  z_post_log_var = self._encoder(input_encoder_1,
                                                     input_encoder_2,
                                                     input_encoder_3,
                                                     batch_size,
                                                     reuse=reuse,
                                                     checkpoint=checkpoint,
                                                     name='teacher')

        self._dist_post = GaussObj(z_post_mean, z_post_log_var, self._nb_latent, batch_size)
        z_post_sample = self._dist_post.sampler()

        y_post_label_logit, y_post_label = self._classifier(z_post_sample,
                                                            batch_size=batch_size,
                                                            checkpoint=checkpoint,
                                                            reuse=reuse,
                                                            name='class-teacher')
        y_post_label_logit_soft = y_post_label_logit/self._temperature
        y_post_label_soft = tf.nn.softmax(y_post_label_logit_soft)
        self._y_post_label_soft = y_post_label_soft

        ### get the parameters of prior dist
        z_prior_mean,  z_prior_log_var = self._encoder(input_encoder_1,
                                                       input_encoder_4,
                                                       input_encoder_5,
                                                       batch_size,
                                                       reuse=reuse,
                                                       checkpoint=None,
                                                       name='student')

        self._dist_prior = GaussObj(z_prior_mean, z_prior_log_var, self._nb_latent, batch_size)
        z_prior_sample = self._dist_prior.sampler()

        y_prior_label_logit, y_prior_label = self._classifier(z_prior_sample,
                                                              batch_size=batch_size,
                                                              checkpoint=None,
                                                              reuse=reuse,
                                                              name='class-student')

        self._y_prior_label_soft_logit = y_prior_label_logit/self._temperature
        self._y_prior_label_true_logit = y_prior_label_logit

        if self._disc_loss:
            self._prob_fake_logits, _ = self._discriminator(z_prior_sample, batch_size, reuse=reuse)
            self._prob_real_logits, _ = self._discriminator(z_post_sample, batch_size, reuse=True)

        return y_prior_label

    def _discriminator(self, z, batch_size, name='disc', reuse=False):

        with tf.variable_scope(name, reuse=reuse):

            is_training = self._is_training
            z = tf.reshape(z, [batch_size, 2, 2, 2, 1])

            current = conv_block(layer_input=z,
                                 nb_kernels=4,
                                 is_training=is_training,
                                 activation=tf.nn.relu,
                                 bn=True,
                                 use_bias=False,
                                 use_dr=False,
                                 down_sampled=False,
                                 checkpoint=None,
                                 scope=name,
                                 name='d9')

            current = tf.reshape(current, [batch_size, -1])

            output_logit = dense3d(layer_input=current,
                                   nb_neurons=1,
                                   is_training=is_training,
                                   activation=None,
                                   use_bias=True,
                                   bn=False,
                                   checkpoint=None,
                                   scope=name,
                                   name='d10')

            print("here output disc:", output_logit.get_shape().as_list())
            output = tf.nn.sigmoid(output_logit)
            return output_logit, output

    def _classifier(self, z, batch_size, name="cnn", checkpoint=None, reuse=False):

        with tf.variable_scope(name, reuse=reuse):

            is_training = self._is_training

            z = tf.reshape(z, [batch_size, -1])

            output_logit = dense3d(layer_input=z,
                                   nb_neurons=self._nb_class,
                                   is_training=is_training,
                                   activation=None,
                                   use_bias=True,
                                   bn=False,
                                   checkpoint=checkpoint,
                                   scope=name,
                                   name='d8')

            output = tf.nn.softmax(output_logit)

            print("here output count:", output.get_shape().as_list())

        return output_logit, output

    def _encoder(self, x1, x2, x3, batch_size, name='encoder', checkpoint=None, reuse=False):

        base_num_filter = self._base_num_filter
        use_dr = self._use_dr
        dr_rate = self._dr_rate
        is_training = self._is_training

        nb_kernels = [base_num_filter, base_num_filter, base_num_filter * 2,
                      base_num_filter * 3, base_num_filter * 2, base_num_filter * 1]

        with tf.variable_scope(name, reuse=reuse):

            current = tf.reshape(x1, [batch_size, ] + self._image_shape)
            if x2 is not None:
                if name == 'teacher':
                    x_2 = tf.reshape(x2, [batch_size, ] + self._image_shape)
                    current = current * x_2
                else:
                    x2 = tf.reshape(x2, [batch_size, ] + self._image_shape[:-1] + [1])
                    current = tf.concat([current, x2], axis=-1)
            if x3 is not None:
                x3 = tf.reshape(x3, [batch_size, ] + self._image_shape[:-1] + [1])
                current = tf.concat([current, x3], axis=-1)
            if not reuse:
                print("encoder input:", current.get_shape().as_list())

            for i in range(len(nb_kernels)):

                bn = False if i == 0 else True
                use_bias = True if i == 0 else False
                down_sampled = False if i == 0 else True
                use_dr = False if i == 0 else use_dr

                current = conv_res_block(layer_input=current,
                                         nb_kernels=nb_kernels[i],
                                         is_training=is_training,
                                         bn=bn,
                                         use_bias=use_bias,
                                         use_dr=use_dr,
                                         dr_rate=dr_rate,
                                         down_sampled=down_sampled,
                                         checkpoint=checkpoint,
                                         scope=name,
                                         name='d' + str(i))

                if not reuse:
                    print("encoder:", current.get_shape().as_list())

            if self._broadcast:
                current = tf.reshape(current, [batch_size, -1])

                z_mu = dense3d(layer_input=current,
                               nb_neurons=self._nb_latent,
                               is_training=is_training,
                               activation=None,
                               use_bias=True,
                               bn=False,
                               checkpoint=checkpoint,
                               scope=name,
                               name='z_mu')

                z_mu = tf.reshape(z_mu, [batch_size, self._nb_latent, 1])

                if not reuse:
                    print("encoder z_mu:", z_mu.get_shape().as_list())

                z_log_sigma = dense3d(layer_input=current,
                                      nb_neurons=self._nb_latent,
                                      is_training=is_training,
                                      activation=None,
                                      use_bias=True,
                                      use_dr=False,
                                      bn=False,
                                      checkpoint=checkpoint,
                                      scope=name,
                                      name='z_log_sigma')

                z_log_sigma = tf.reshape(z_log_sigma, [batch_size, self._nb_latent, 1])

                if not reuse:
                    print("encoder z_log_sigma:", z_log_sigma.get_shape().as_list())

            else:
                z_mu = conv3d(layer_input=current,
                              nb_kernels=1,
                              kernel_size=(1, 1, 1),
                              bn=False,
                              use_bias=True,
                              activation=None,
                              is_training=is_training,
                              checkpoint=checkpoint,
                              scope=name,
                              name='z_mu')

                z_mu = tf.reshape(z_mu, [batch_size, self._nb_latent, 1])

                if not reuse:
                    print("encoder z_mu:", z_mu.get_shape().as_list())

                z_log_sigma = conv3d(layer_input=current,
                                     nb_kernels=1,
                                     kernel_size=(1, 1, 1),
                                     bn=False,
                                     use_bias=True,
                                     activation=False,
                                     is_training=is_training,
                                     checkpoint=checkpoint,
                                     scope=name,
                                     name='z_log_sigma')

                z_log_sigma = tf.reshape(z_log_sigma, [batch_size, self._nb_latent, 1])

                if not reuse:
                    print("encoder z_log_sigma:", z_log_sigma.get_shape().as_list())

        return z_mu, z_log_sigma

    def _optimizer(self, total_loss, global_step):

        lr = tf.train.exponential_decay(learning_rate=self._lr,
                                        global_step=global_step,
                                        decay_steps=10000,
                                        decay_rate=.9999,
                                        staircase=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        teacher_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="teacher/")
        teacher_class_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="class-teacher/")
        student_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="student/")
        student_class_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="class-student/")
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="disc/")

        student_loss = 0
        for loss_name in total_loss.keys():
            if loss_name != 'loss(disc)':
                student_loss += total_loss[loss_name]
        if self._use_disc_loss:
            disc_loss = total_loss['loss(disc)']

        var_list = student_class_vars + student_vars

        with tf.control_dependencies(update_ops):
            optimiser = tf.train.AdamOptimizer(learning_rate=lr)
            gradients = optimiser.compute_gradients(student_loss, var_list=var_list)
            solver = optimiser.apply_gradients(gradients, global_step=global_step)

            # to add to tensorboard
            tf.summary.scalar('lr', lr)
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            for gradient, variable in gradients:
                tf.summary.histogram("gradients/" + variable.name, l2_norm(gradient))
                tf.summary.histogram("variables/" + variable.name, l2_norm(variable))

            if self._use_disc_loss:
                optimiser_disc = tf.train.AdamOptimizer(learning_rate=lr * 0.1)
                gradients_disc = optimiser_disc.compute_gradients(disc_loss, var_list=disc_vars)
                solver_adv = optimiser_disc.apply_gradients(gradients_disc, global_step=global_step)

                l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
                for gradient, variable in gradients_disc:
                    tf.summary.histogram("gradients/" + variable.name, l2_norm(gradient))
                    tf.summary.histogram("variables/" + variable.name, l2_norm(variable))
                return solver, solver_adv

        return solver

    def _kl_loss(self, coef=1, name='loss(kl)'):

        # tensorflow built-in kl loss
        kl_loss = (1/self._nb_latent) * tfp.distributions.kl_divergence(self._dist_post.get_dist(), self._dist_prior.get_dist()) #

        # my implementation of kl loss
        z_prior_mean = self._dist_prior.get_mean()
        z_prior_log_var = self._dist_prior.get_log_var()
        z_post_mean = self._dist_post.get_mean()
        z_post_log_var = self._dist_post.get_log_var()
        kl_loss_2 = -0.5 * tf.reduce_mean(1 + z_post_log_var - z_prior_log_var - (
                tf.exp(z_post_log_var) + (z_post_mean - z_prior_mean) ** 2) / tf.exp(z_prior_log_var),
                                        axis=[1, 2])

        self._tot_loss[name] = tf.reduce_mean(kl_loss, axis=0)
        self._coef_loss[name] = coef
        #tf.summary.scalar(name + '-mine', tf.reduce_mean(coef * kl_loss_2))
        #tf.summary.scalar(name, tf.reduce_mean(coef * kl_loss))
        #tf.summary.scalar('beta', coef)

    def _CE_loss(self, logits, labels, true_labels, soft_label=False, coef=1., name='loss(class)'):

        class_weights = self._class_weight

        # weighted cross entropy loss
        y_true_one_hot = tf.one_hot(true_labels, self._nb_class)
        weights = tf.reduce_sum(class_weights * y_true_one_hot, axis=1)

        if soft_label:
            unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        else:
            unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        loss_label = unweighted_loss * weights

        self._tot_loss[name] = tf.reduce_mean(loss_label, axis=0)
        self._coef_loss[name] = coef
        #tf.summary.scalar(name, tf.reduce_mean(loss_label))

    def _disc_loss(self, coef=1., name='loss(disc)'):
        prob_fake_logits = self._prob_fake_logits
        prob_real_logits = self._prob_real_logits
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prob_real_logits, labels=tf.ones_like(prob_real_logits)),
            axis=[1])
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prob_fake_logits, labels=tf.zeros_like(prob_fake_logits)),
            axis=[1])
        disc_loss = real_loss + fake_loss

        self._tot_loss[name] = tf.reduce_mean(disc_loss, axis=0)
        self._coef_loss[name] = coef
        #tf.summary.scalar(name, tf.reduce_mean(coef * disc_loss))
        return disc_loss

    def _grl_loss(self, coef=1., name='loss(grl)'):
        grl_loss = -1 * self._disc_loss()
        self._tot_loss[name] = tf.reduce_mean(grl_loss, axis=0)
        self._coef_loss[name] = coef
        #tf.summary.scalar(name, tf.reduce_mean(grl_loss))

    def _dc_loss(self, coef=1., name='loss(dc)'):
        prob_fake_logits = self._prob_fake_logits
        dc_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=prob_fake_logits, labels=tf.ones_like(prob_fake_logits)),
            axis=[1])
        self._tot_loss[name] = tf.reduce_mean(dc_loss, axis=0)
        self._coef_loss[name] = coef
        #tf.summary.scalar(name, tf.reduce_mean(coef * dc_loss))

    def _loss(self):

        beta = self._beta

        self._tot_loss = {}
        self._coef_loss = {}

        y_true_label = self._y_true_label
        y_prior_labels_true_logit = self._y_prior_label_true_logit

        y_soft_label = self._y_post_label_soft
        y_prior_labels_soft_logit = self._y_prior_label_soft_logit

        # weighted cross entropy loss for true labels
        self._CE_loss(logits=y_prior_labels_true_logit,
                      labels=y_true_label,
                      true_labels=y_true_label,
                      soft_label=False,
                      coef=0.1,
                      name='loss(class-true)')

        # weighted cross entropy loss for soft labels
        if self._use_soft_label:
            self._CE_loss(logits=y_prior_labels_soft_logit,
                          labels=y_soft_label,
                          true_labels=y_true_label,
                          soft_label=True,
                          coef=0.9,
                          name='loss(class-soft)')

        if self._use_kl_loss:
            self._kl_loss(coef=beta)

        if self._use_disc_loss:
            self._disc_loss()
            if self._use_grl_loss:
                self._grl_loss()
            else:
                self._dc_loss()

        if self._use_activity_loss:
            self._activity_loss()

        loss = {}
        for loss_name in self._tot_loss.keys():
            loss[loss_name] = self._coef_loss[loss_name] * self._tot_loss[loss_name]

        return loss

    def get_loss(self):
        return self._loss()

    def get_optimizer(self, loss, global_step):
        return self._optimizer(loss, global_step)

    def get_encoder(self):
        return self._encoder

    def get_classifier(self):
        return self._classifier

    def get_discriminator(self):
        return self._discriminator
