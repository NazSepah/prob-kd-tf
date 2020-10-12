import tensorflow as tf

_DATA_FORMAT = 'channels_last'
_DATA_FORMAT_NN = "NDHWC"

xavier_uniform_initilizer = tf.glorot_uniform_initializer(seed=1500)
xavier_normal_initilizer = tf.glorot_normal_initializer(seed=1500)
truncated_normal_initializer = tf.initializers.truncated_normal(mean=0., stddev=0.001)
_KERNEL_INIT = xavier_normal_initilizer

_BN_CODE = {"c0":'', "c1":"_1", "c2":"_2"}

_BN_CODE_0 = {"c0":'', "c1":"", "c2":"_1"}


def dense3d(layer_input,
            nb_neurons,
            activation=tf.nn.relu,
            bn=True,
            is_training=True,
            use_bias=False,
            use_dr=False,
            dr_rate=0.2,
            reuse=False,
            checkpoint=None,
            scope=None,
            name=None):

        x = layer_input

        x = tf.layers.dropout(inputs=x,
                              rate=dr_rate,
                              training=use_dr)
        if checkpoint is not None:
            kernel_init = tf.constant_initializer(
                tf.train.load_variable(checkpoint, scope + '/' + name + '/kernel'))

        else:
            kernel_init = _KERNEL_INIT

        if checkpoint is not None and use_bias:
            bias_init = tf.constant_initializer(
                    tf.train.load_variable(checkpoint, scope + '/' + name + '/bias'))
        else:
            bias_init = tf.zeros_initializer()

        x = tf.layers.dense(inputs=x,
                            units=nb_neurons,
                            activation=None,
                            use_bias=use_bias,
                            kernel_initializer=kernel_init,
                            bias_initializer=bias_init,
                            name=name)

        if bn:

            x = tf.layers.batch_normalization(inputs=x,
                                              training=is_training,
                                              reuse=reuse)

        if activation:
            x = activation(x)
        return x


def conv3d(layer_input,
           nb_kernels,
           kernel_size=(3, 3, 3),
           strides=(1, 1, 1),
           activation=tf.nn.relu,
           kernel_regularizer=None, #tf.contrib.layers.l2_regularizer(0.01),
           bn=True,
           padding='SAME',
           is_training=True,
           use_bias=False,
           batch_norm=False,
           reuse=False,
           checkpoint=None,
           scope=None,
           name=None):

    x = layer_input

    if checkpoint is not None:
        kernel_init = tf.constant_initializer(tf.train.load_variable(checkpoint, scope + '/' + name + '/kernel'))

        try:
            if int(scope[-1]) == 0:
                bn_code = _BN_CODE_0
            else:
                bn_code = _BN_CODE
        except ValueError:
            bn_code = _BN_CODE

    else:
        kernel_init = _KERNEL_INIT

    if checkpoint is not None and use_bias:
        bias_init = tf.constant_initializer(
            tf.train.load_variable(checkpoint, scope + '/' + name + '/bias'))
    else:
        bias_init = tf.zeros_initializer()

    if checkpoint is not None and bn and ~batch_norm:
        instance_norme_beta_init = tf.constant_initializer(tf.train.load_variable(checkpoint, scope + '/InstanceNorm{}/beta'.format(bn_code[name])))
        instance_norme_gamma_init = tf.constant_initializer(tf.train.load_variable(checkpoint, scope + '/InstanceNorm{}/gamma'.format(bn_code[name])))

    else:
        instance_norme_beta_init = tf.zeros_initializer()
        instance_norme_gamma_init = tf.ones_initializer()

    x = tf.layers.conv3d(inputs=x,
                         filters=nb_kernels,
                         kernel_size=kernel_size,
                         strides=strides,
                         data_format=_DATA_FORMAT,
                         activation=None,
                         use_bias=use_bias,
                         kernel_initializer=kernel_init,
                         bias_initializer=bias_init,
                         kernel_regularizer=kernel_regularizer,
                         padding=padding,
                         reuse=reuse,
                         name=name)

    if bn:
        if batch_norm:
            x = tf.layers.batch_normalization(inputs=x,
                                              training=is_training,
                                              reuse=reuse)
        else:
            x = tf.contrib.layers.instance_norm(inputs=x,
                                                param_initializers={'beta': instance_norme_beta_init,
                                                                    "gamma": instance_norme_gamma_init})

    if activation:
        x = activation(x)
    return x


def conv_res_block(layer_input,
                   nb_kernels,
                   kernel_size=(3, 3, 3),
                   strides=(2, 2, 2),
                   down_sampled=True,
                   activation=tf.nn.relu,
                   kernel_regularizer=None, #tf.contrib.layers.l2_regularizer(0.01),
                   bn=True,
                   pooling=True,
                   use_dr=True,
                   dr_rate=0.1,
                   is_training=True,
                   use_bias=False,
                   checkpoint=None,
                   reuse=False,
                   scope=None,
                   name=None):

    with tf.variable_scope("layer_" + name):

        nb_current_kernels = layer_input.get_shape().as_list()[-1]

        x = tf.layers.dropout(inputs=layer_input,
                              rate=dr_rate,
                              training=use_dr)

        x = conv3d(layer_input=x,
                   nb_kernels=nb_kernels,
                   kernel_size=kernel_size,
                   activation=activation,
                   kernel_regularizer=kernel_regularizer,
                   bn=bn,
                   is_training=is_training,
                   use_bias=use_bias,
                   reuse=reuse,
                   checkpoint=checkpoint,
                   scope=scope + '/layer_' + name,
                   name="c0")

        x = conv3d(layer_input=x,
                   nb_kernels=nb_kernels,
                   kernel_size=kernel_size,
                   activation=activation,
                   kernel_regularizer=kernel_regularizer,
                   bn=True,
                   is_training=is_training,
                   use_bias=use_bias,
                   reuse=reuse,
                   checkpoint=checkpoint,
                   scope=scope + '/layer_' + name,
                   name="c1")

        if nb_kernels != nb_current_kernels:

            skip = conv3d(layer_input=layer_input,
                          nb_kernels=nb_kernels,
                          kernel_size=(1, 1, 1),
                          strides=(1, 1, 1),
                          activation=activation,
                          kernel_regularizer=kernel_regularizer,
                          bn=True,
                          is_training=is_training,
                          use_bias=use_bias,
                          reuse=reuse,
                          checkpoint=checkpoint,
                          scope=scope + '/layer_' + name,
                          name="c2")
        else:
            skip = layer_input
        x = x + skip

        if down_sampled:
            if pooling:
                x = tf.layers.max_pooling3d(inputs=x,
                                            pool_size=(3, 3, 3),
                                            strides=strides,
                                            padding='same')
            else:
                x = conv3d(layer_input=x,
                           nb_kernels=nb_kernels,
                           kernel_size=kernel_size,
                           strides=strides,
                           activation=activation,
                           bn=True,
                           is_training=is_training,
                           use_bias=use_bias,
                           reuse=reuse,
                           checkpoint=checkpoint,
                           scope=scope + '/layer_' + name,
                           name="c3")

    return x


def conv_block(layer_input,
               nb_kernels,
               kernel_size=(3, 3, 3),
               strides=(2, 2, 2),
               down_sampled=True,
               activation=tf.nn.relu,
               kernel_regularizer=None, #tf.contrib.layers.l2_regularizer(0.01),
               bn=True,
               use_dr=False,
               dr_rate=0.2,
               is_training=True,
               use_bias=False,
               reuse=False,
               checkpoint=None,
               scope=None,
               name=None):
    x = layer_input

    with tf.variable_scope("layer_" + name):

        x = tf.layers.dropout(inputs=x,
                              rate=dr_rate,
                              training=use_dr)

        x = conv3d(layer_input=x,
                   nb_kernels=nb_kernels,
                   kernel_size=kernel_size,
                   activation=activation,
                   kernel_regularizer=kernel_regularizer,
                   bn=bn,
                   is_training=is_training,
                   use_bias=use_bias,
                   checkpoint=checkpoint,
                   reuse=reuse,
                   scope=scope + '/layer_' + name,
                   name="c0")

        x = conv3d(layer_input=x,
                   nb_kernels=nb_kernels,
                   kernel_size=kernel_size,
                   activation=activation,
                   kernel_regularizer=kernel_regularizer,
                   bn=True,
                   is_training=is_training,
                   use_bias=use_bias,
                   checkpoint=checkpoint,
                   scope=scope + '/layer_' + name,
                   reuse=reuse,
                   name="c1")

        if down_sampled:
            x = tf.layers.max_pooling3d(inputs=x,
                                        pool_size=(3, 3, 3),
                                        strides=strides,
                                        padding='same')
    return x

