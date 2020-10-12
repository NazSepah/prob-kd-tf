import tensorflow as tf
import os, json, argparse
from os.path import join
import numpy as np
from shutil import copy


from src.data_source.volume_generator_tfrecord import BrainDataProvider as tfDataProvider
from src.models.cnn_counts_student import CNN
from src.callbacks.callbacks_cnn import ACC, UNC, Dist, DistZ, DISC

_DECAY_RATE = 0.95


def _get_cfg():
    parser = argparse.ArgumentParser(description="Main handler for training",
                                     usage="./script/train.sh -j configs/train.json -g 02")
    parser.add_argument("-j", "--json", help="configuration json file", required=True)
    parser.add_argument('-g', '--gpu', help='Cuda Visible Devices', required=True)
    args = parser.parse_args()
    return args


def _main(args):
    with open(args.json, 'r') as f:
        cfg = json.loads(f.read())

    expt_cfg = cfg['experiment']
    expt_name = expt_cfg['name']
    prob = expt_cfg['prob']
    outdir = expt_cfg['outdir']
    tfdir = expt_cfg['tfdir']
    nb_epochs = expt_cfg['nb_epochs']
    pre_trained = expt_cfg['pre_trained']
    pre_trained_checkpoint = expt_cfg['pre_trained_checkpoint']
    use_disc_loss = cfg['train']["disc_loss"]
    use_kl_loss = cfg['train']["kl_loss"]
    use_soft_label = cfg['train']["soft_label"]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(join(outdir, expt_name), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'graphs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'checkpoints'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'rocs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'recons'), exist_ok=True)
    copy(args.json, join(outdir, expt_name))
    copy("/cim/nazsepah/projects/kd3d-tf/src/models/cnn_counts_student.py", join(outdir, expt_name))

    gen_train = tfDataProvider(tfdir, cfg['train'])
    gen_valid = tfDataProvider(tfdir, cfg['valid'])
    cfg['callback']['mode'] = 'train'
    gen_callback_train = tfDataProvider(tfdir, cfg['callback'])
    cfg['callback']['mode'] = 'valid'
    gen_callback_valid = tfDataProvider(tfdir, cfg['callback'])

    train_data = gen_train.data_generator()
    valid_data = gen_valid.data_generator()
    callback_train_data = gen_callback_train.data_generator()
    callback_valid_data = gen_callback_valid.data_generator()

    nb_samples_train = gen_train.get_nb_samples()
    nb_samples_valid = gen_valid.get_nb_samples()
    nb_samples_callback_train = gen_callback_train.get_nb_samples()
    nb_samples_callback_valid = gen_callback_valid.get_nb_samples()

    print("total number of training samples:", nb_samples_train)
    print("total number of validation samples:", nb_samples_valid)
    print("total number of callback samples:", nb_samples_callback_valid)

    nb_batches_train = int(np.ceil(nb_samples_train / cfg['train']['batch_size']))
    nb_batches_valid = int(np.ceil(nb_samples_valid / cfg['valid']['batch_size']))
    nb_batches_callback_train = int(np.ceil(nb_samples_callback_train / cfg['callback']['batch_size']))
    nb_batches_callback_valid = int(np.ceil(nb_samples_callback_valid / cfg['callback']['batch_size']))

    print("nb batch train:", nb_batches_train)

    iterator = tf.data.Iterator.from_structure(train_data.output_types)
    train_init_op = iterator.make_initializer(train_data)
    valid_init_op = iterator.make_initializer(valid_data)
    next_batch = iterator.get_next()

    # setup the model, loss and metrics
    class_weights = tf.placeholder(dtype=tf.float32, shape=(1, cfg['train']['nb_class']), name='loss_weight')
    beta = tf.placeholder_with_default(1., shape=(), name='beta')
    mode = tf.placeholder(tf.bool, name='mode')
    is_training = tf.placeholder(tf.bool, name='is_training')
    use_dr = tf.placeholder(tf.bool, name='is_dr')
    if pre_trained:
        checkpoint = join(outdir, pre_trained_checkpoint)
    else:
        checkpoint = None
    cnn = CNN(cfg['train'], is_training, use_dr)

    pred = cnn.build_model(next_batch, class_weights, beta, cfg['train']['batch_size'],  mode, checkpoint=checkpoint)
    loss = cnn.get_loss()

    # set up the optimizer
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0), trainable=False)

    solver = cnn.get_optimizer(loss, global_step)
    if use_disc_loss:
        solver, solver_disc = solver

    loss_ce_true = tf.placeholder(dtype=tf.float32, shape=(), name='loss_ce_true')
    loss_ce_soft = tf.placeholder(dtype=tf.float32, shape=(), name='loss_ce_soft')

    loss_disc = tf.placeholder(dtype=tf.float32, shape=(), name='loss_disc')
    loss_dc = tf.placeholder(dtype=tf.float32, shape=(), name='loss_dc')

    loss_kl = tf.placeholder(dtype=tf.float32, shape=(), name='loss_kl')
    loss_all = tf.placeholder(dtype=tf.float32, shape=(), name='loss')

    loss_ce_true_tf = tf.summary.scalar('avg_loss_ce_true', loss_ce_true)
    loss_ce_soft_tf = tf.summary.scalar('avg_loss_ce_soft', loss_ce_soft)

    loss_disc_tf = tf.summary.scalar('avg_loss_disc', loss_disc)
    loss_dc_tf = tf.summary.scalar('avg_loss_dc', loss_dc)

    loss_kl_tf = tf.summary.scalar('avg_loss_kl', loss_kl)
    loss_tf = tf.summary.scalar('avg_loss', loss_all)

    avg_loss_summary = tf.summary.merge([loss_ce_true_tf, loss_ce_soft_tf, loss_disc_tf, loss_dc_tf, loss_kl_tf, loss_tf])

    # set-up the callbacks
    acc_1 = tf.placeholder(dtype=tf.float32, shape=(), name='acc_1')
    acc_2 = tf.placeholder(dtype=tf.float32, shape=(), name='acc_2')

    disc_fake = tf.placeholder(dtype=tf.float32, shape=(), name='disc_fake')
    disc_real = tf.placeholder(dtype=tf.float32, shape=(), name='disc_real')

    acc_tf_1 = tf.summary.scalar('acc_1', acc_1)
    acc_tf_2 = tf.summary.scalar('acc_2', acc_2)

    disc_fake_tf = tf.summary.scalar('disc_output_fake', disc_fake)
    disc_real_tf = tf.summary.scalar('disc_output_real', disc_real)

    acc_callback = ACC(cfg, reuse=True)
    disc_callback = DISC(cfg, reuse=True)

    if prob:
        unc_1 = tf.placeholder(dtype=tf.float32, shape=(), name='unc_correct')
        unc_2 = tf.placeholder(dtype=tf.float32, shape=(), name='unc_wrong')

        dist_y = tf.placeholder(dtype=tf.float32, shape=(), name='dist_y')
        dist_z_point = tf.placeholder(dtype=tf.float32, shape=(), name='distz_point')
        dist_z_post_bc = tf.placeholder(dtype=tf.float32, shape=(), name='distz_post_bc')
        dist_z_post_wc = tf.placeholder(dtype=tf.float32, shape=(), name='distz_post_wc')
        dist_z_prior_bc = tf.placeholder(dtype=tf.float32, shape=(), name='distz_prior_bc')
        dist_z_prior_wc = tf.placeholder(dtype=tf.float32, shape=(), name='distz_prior_wc')

        unc_tf_1 = tf.summary.scalar('unc_correct', unc_1)
        unc_tf_2 = tf.summary.scalar('unc_wrong', unc_2)

        distz_point_tf = tf.summary.scalar('dist_z_point', dist_z_point)
        distz_post_bc_tf = tf.summary.scalar('dist_z_post_bc', dist_z_post_bc)
        distz_post_wc_tf = tf.summary.scalar('dist_z_post_wc', dist_z_post_wc)
        distz_prior_bc_tf = tf.summary.scalar('dist_z_prior_bc', dist_z_prior_bc)
        distz_prior_wc_tf = tf.summary.scalar('dist_z_prior_wc', dist_z_prior_wc)
        disty_tf = tf.summary.scalar('dist_y', dist_y)

        unc_callback = UNC(cfg, reuse=True)
        dist_callback = Dist(cfg, reuse=True)
        distZ_callback = DistZ(cfg, reuse=True)

        merged_summary_callback = tf.summary.merge([acc_tf_1, acc_tf_2, unc_tf_1, unc_tf_2, disty_tf,
                                                    distz_point_tf, distz_post_bc_tf, distz_post_wc_tf,
                                                    distz_prior_bc_tf, distz_prior_wc_tf,
                                                    disc_real_tf, disc_fake_tf])
    else:
        merged_summary_callback = tf.summary.merge([acc_tf_1, acc_tf_2,  disc_real_tf, disc_fake_tf])

    saver = tf.train.Saver(max_to_keep=50)
    init_op = tf.global_variables_initializer()

    sess_config = tf.ConfigProto(intra_op_parallelism_threads=8,
                                 inter_op_parallelism_threads=8,
                                 device_count={'CPU': 8})

    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)
        sess.run(train_init_op)
        #saver.restore(sess, checkpoint)

        train_writer = tf.summary.FileWriter(join(outdir, expt_name, 'graphs', 'train'), sess.graph)
        valid_writer = tf.summary.FileWriter(join(outdir, expt_name, 'graphs', 'valid'), sess.graph)

        print("training starts!!!")

        for epoch in range(nb_epochs):
            print("Epoch {}/{}".format(epoch, nb_epochs), flush=True)

            ##### training ###########
            sess.run(train_init_op)

            beta_value = 1. #* (1 - _DECAY_RATE ** (epoch - 5.)) if epoch > 5. else 0.0
            class_weights_value = np.array([[2.,  1.]])
            mode_value = False

            train_loss = 0.
            loss_ce_soft_np = 0.
            loss_ce_true_np = 0.
            loss_disc_np = 0.
            loss_dc_np = 0.
            loss_kl_np = 0.

            for itr in range(2 * nb_batches_train):

                b, p, l, _ = sess.run([next_batch, pred, loss, solver], feed_dict={is_training: True,
                                                                                   use_dr: False,
                                                                                   beta: beta_value,
                                                                                   mode: mode_value,
                                                                                   class_weights: class_weights_value})

                loss_ce_true_np += l['loss(class-true)']
                loss_kl_np += l['loss(kl)'] if use_kl_loss else 0.
                loss_dc_np += l['loss(dc)'] if use_disc_loss else 0.
                loss_ce_soft_np += l['loss(class-soft)'] if use_soft_label else 0.

                if use_disc_loss:
                    l, _ = sess.run([loss, solver_disc], feed_dict={is_training: True,
                                                                    use_dr: False,
                                                                    beta: beta_value,
                                                                    mode: mode_value,
                                                                    class_weights: class_weights_value})
                    loss_disc_np += l['loss(disc)']

                for key in l.keys():
                    train_loss += l[key]

            loss_ce_soft_np = loss_ce_soft_np / nb_batches_train
            loss_ce_true_np = loss_ce_true_np / nb_batches_train
            loss_disc_np = loss_disc_np / nb_batches_train
            loss_dc_np = loss_dc_np / nb_batches_train
            loss_kl_np = loss_kl_np / nb_batches_train
            train_loss = train_loss / nb_batches_train

            summary = sess.run(avg_loss_summary, feed_dict={loss_ce_soft: loss_ce_soft_np,
                                                            loss_ce_true: loss_ce_true_np,
                                                            loss_disc: loss_disc_np,
                                                            loss_dc: loss_dc_np,
                                                            loss_kl: loss_kl_np,
                                                            loss_all: train_loss})

            train_writer.add_summary(summary, global_step=epoch)

            #### callbacks ####
            if epoch % 5 == 0:
                saver.save(sess, join(outdir, expt_name, 'checkpoints', 'model.ckpt-{}'.format(epoch)))

                acc_value = acc_callback.on_epoch_end(callback_data=callback_train_data,
                                                      nb_imgs=nb_batches_callback_train,
                                                      mode_value=mode_value,
                                                      epoch=epoch)

                disc_value = disc_callback.on_epoch_end(callback_data=callback_train_data,
                                                        nb_imgs=nb_batches_callback_train,
                                                        epoch=epoch)
                if prob:
                    dist_value = dist_callback.on_epoch_end(callback_data=callback_train_data,
                                                            nb_imgs=nb_batches_callback_train,
                                                            epoch=epoch)

                    unc_value = unc_callback.on_epoch_end(callback_data=callback_train_data,
                                                          nb_imgs=nb_batches_callback_train,
                                                          nb_samples=10,
                                                          epoch=epoch)

                    distz_value = distZ_callback.on_epoch_end(callback_data=callback_train_data,
                                                              nb_imgs=nb_batches_callback_train,
                                                              epoch=epoch)

                    summary = sess.run(merged_summary_callback, feed_dict={acc_1: acc_value[0],
                                                                           acc_2: acc_value[1],
                                                                           disc_real: disc_value[0],
                                                                           disc_fake: disc_value[1],
                                                                           unc_1: unc_value[0],
                                                                           unc_2: unc_value[1],
                                                                           dist_z_point: dist_value[0],
                                                                           dist_y: dist_value[1],
                                                                           dist_z_post_bc: distz_value[0],
                                                                           dist_z_post_wc: distz_value[1],
                                                                           dist_z_prior_bc: distz_value[2],
                                                                           dist_z_prior_wc: distz_value[3],
                                                                           })
                    train_writer.add_summary(summary, global_step=epoch)
                    
                else:
                    summary = sess.run(merged_summary_callback, feed_dict={acc_1: acc_value[0],
                                                                           acc_2: acc_value[1],
                                                                           disc_real: disc_value[0],
                                                                           disc_fake: disc_value[1],
                                                                           })
                    train_writer.add_summary(summary, global_step=epoch)


            ##### validation ##########
            sess.run(valid_init_op)
            beta_value = 1.
            class_weights_value = np.array([[1., 1.]])
            mode_value = False

            valid_loss = 0.
            loss_ce_soft_np = 0.
            loss_ce_true_np = 0.
            loss_disc_np = 0.
            loss_dc_np = 0.
            loss_kl_np = 0.

            for itr in range(nb_batches_valid):
                l = sess.run(loss, feed_dict={is_training: False, use_dr: False, beta: beta_value,
                                              mode: mode_value, class_weights: class_weights_value})

                loss_ce_true_np += l['loss(class-true)']
                loss_ce_soft_np += l['loss(class-soft)'] if use_soft_label else 0.
                loss_kl_np += l['loss(kl)'] if use_kl_loss else 0.
                loss_disc_np += l['loss(disc)'] if use_disc_loss else 0.
                loss_dc_np += l['loss(dc)'] if use_disc_loss else 0.

                for key in l.keys():
                    valid_loss += l[key]

            loss_ce_soft_np = loss_ce_soft_np / nb_batches_valid
            loss_ce_true_np = loss_ce_true_np / nb_batches_valid
            loss_disc_np = loss_disc_np / nb_batches_valid
            loss_dc_np = loss_dc_np / nb_batches_valid
            loss_kl_np = loss_kl_np / nb_batches_valid
            valid_loss = valid_loss / nb_batches_valid

            summary = sess.run(avg_loss_summary, feed_dict={loss_ce_soft: loss_ce_soft_np,
                                                            loss_ce_true: loss_ce_true_np,
                                                            loss_disc: loss_disc_np,
                                                            loss_dc: loss_dc_np,
                                                            loss_kl: loss_kl_np,
                                                            loss_all: valid_loss})

            valid_writer.add_summary(summary, global_step=epoch)

            print("train loss: {:03.2f}, valid loss: {:03.2f}".format(train_loss, valid_loss), flush=True)

            #### callbacks ####
            if epoch % 5 == 0:
                acc_value = acc_callback.on_epoch_end(callback_data=callback_valid_data,
                                                      nb_imgs=nb_batches_callback_valid,
                                                      mode_value=mode_value,
                                                      epoch=epoch)

                disc_value = disc_callback.on_epoch_end(callback_data=callback_valid_data,
                                                        nb_imgs=nb_batches_callback_valid,
                                                        epoch=epoch)

                if prob:
                    dist_value = dist_callback.on_epoch_end(callback_data=callback_valid_data,
                                                            nb_imgs=nb_batches_callback_valid,
                                                            epoch=epoch)

                    unc_value = unc_callback.on_epoch_end(callback_data=callback_valid_data,
                                                          nb_imgs=nb_batches_callback_valid,
                                                          nb_samples=10,
                                                          epoch=epoch)

                    distz_value = distZ_callback.on_epoch_end(callback_data=callback_valid_data,
                                                              nb_imgs=nb_batches_callback_valid,
                                                              epoch=epoch)

                    summary = sess.run(merged_summary_callback, feed_dict={acc_1: acc_value[0],
                                                                           acc_2: acc_value[1],
                                                                           disc_real: disc_value[0],
                                                                           disc_fake: disc_value[1],
                                                                           unc_1: unc_value[0],
                                                                           unc_2: unc_value[1],
                                                                           dist_z_point: dist_value[0],
                                                                           dist_y: dist_value[1],
                                                                           dist_z_post_bc: distz_value[0],
                                                                           dist_z_post_wc: distz_value[1],
                                                                           dist_z_prior_bc: distz_value[2],
                                                                           dist_z_prior_wc: distz_value[3],
                                                                           })
                    valid_writer.add_summary(summary, global_step=epoch)
                else:
                    summary = sess.run(merged_summary_callback, feed_dict={acc_1: acc_value[0],
                                                                           acc_2: acc_value[1],
                                                                           disc_real: disc_value[0],
                                                                           disc_fake: disc_value[1],
                                                                           })
                    valid_writer.add_summary(summary, global_step=epoch)


if __name__ == "__main__":
    _main(_get_cfg())
