import tensorflow as tf
from os.path import join
import numpy as np
import nibabel as nib
from data_reader_placebo import Reader
import argparse
import json


_MASK_PATH = {'MS-LAQ-302-STX': r'/cim/data/neurorx/MS-LAQ-302-STX/extra/anewt2_75/',
              '101MS326': r'/cim/data/neurorx/lesion_files/'}


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _pre_process(_id, writer_params, time_points, dtype=np.float32):
    input_mods = writer_params['input_mods']
    output_mods = writer_params['output_mods']
    dim = writer_params['dim']
    bbox = writer_params['bbox']

    nb_mod_input = len(list(input_mods.keys()))
    nb_mod_output = len(list(output_mods.keys()))

    input_imgs = np.empty((nb_mod_input + 2, *dim), dtype=dtype)
    output_imgs = np.empty((nb_mod_output + 1, *dim), dtype=dtype)

    subj, tp, subject_folder, dataset, site, img_tag, mask_tag = _id
    tp_indx = time_points.index(tp)
    next_tp = time_points[tp_indx + 1]
    imgs_folder_path = join(subject_folder, tp)
    next_imgs_folder_path = join(subject_folder, next_tp)
    mask_folder_path = join(subject_folder, 'stx152lsq6')

    mask_name = '{}_{}_{}_{}'.format(dataset, site, subj, mask_tag)
    mask_path = join(mask_folder_path, mask_name)
    mask = _centre(nib.load(mask_path).get_data(), bbox)

    for mod in input_mods.keys():
        mod_indx = input_mods[mod]
        img_name = '{}_{}_{}_{}_{}_{}'.format(dataset, site, subj, tp, mod, img_tag)
        img_path = join(imgs_folder_path, img_name)
        img = nib.load(img_path).get_data()
        img = _centre(_normalize(img), bbox) * mask
        input_imgs[mod_indx, ...] = img
    inp_name = img_name
    print("input:", inp_name)

    img_name = '{}_{}_{}_{}_{}_{}'.format(dataset, site, subj, tp, 'ct2f', img_tag)
    img_path = join(imgs_folder_path, img_name)
    img = nib.load(img_path).get_data()
    img = _centre(img, bbox) * mask
    input_imgs[nb_mod_input, ...] = img
    print("input:", img_name)

    img_name = '{}_{}_{}_{}_{}_{}'.format(dataset, site, subj, tp, 'gvf', img_tag)
    img_path = join(imgs_folder_path, img_name)
    img = nib.load(img_path).get_data()
    img = _centre(img, bbox) * mask
    nb_gad = len(np.unique(img)) - 1
    img = (img > 0).astype(int)
    input_imgs[nb_mod_input + 1, ...] = img
    print("input:", img_name)

    for mod in output_mods.keys():
        mod_indx = output_mods[mod]
        img_name = '{}_{}_{}_{}_{}_{}'.format(dataset, site, subj, next_tp, mod, img_tag)
        img_path = join(next_imgs_folder_path, img_name)
        img = nib.load(img_path).get_data()
        img = _centre(_normalize(img), bbox) * mask
        output_imgs[mod_indx, ...] = img
    out_name = img_name
    print("output:", out_name)

    if dataset == 'MS-LAQ-302-STX':
        img_name = '{}_{}_{}_{}_anewt2_75_TREF-{}.mnc.gz'.format(dataset, site, subj, next_tp, tp)
        img_path = join(_MASK_PATH[dataset], img_name)
    elif dataset == '101MS326':
        img_name = '{}_{}_{}_{}_newt2f_TREF-{}_ISPC-stx152lsq6.mnc.gz'.format(dataset, site, subj, next_tp, tp)
        img_path = join(_MASK_PATH[dataset], subject_folder, next_tp, img_name)
    img = nib.load(img_path).get_data()
    img = _centre(img, bbox) * mask
    nb_newt2 = len(np.unique(img)) - 1
    img_newt2 = (img > 0).astype(int)
    output_imgs[nb_mod_output, ...] = img_newt2
    print("output:", img_name)

    input_imgs = _process(input_imgs)
    output_imgs = _process(output_imgs)

    print("input:", np.shape(input_imgs), np.min(input_imgs), np.max(input_imgs))
    print("output:", np.shape(output_imgs), np.min(output_imgs), np.max(output_imgs))
    print("========================")

    return input_imgs, output_imgs, nb_newt2, nb_gad


def _process(x, clip=True):
    x = x.transpose(2, 3, 1, 0)
    x = x[0:-6, 0:-6, :, :]
    if clip:
        x = np.clip(x, 0., 1.)
    x = np.pad(x, ((0, 0), (0, 0), (6, 7), (0, 0)), 'constant', constant_values=0)
    return x


def _normalize(raw_data):
    if np.sum(raw_data) > 0:
        mask = raw_data > 0
        mu = raw_data[mask].mean()
        sigma = raw_data[mask].std()
        data = (raw_data - mu) / (sigma + 0.0001)
        data = np.clip(data, np.min(data), 3)
        data = (data + (-np.min(data))) / (np.minimum(3, np.max(data)) - np.min(data))
        return data
    else:
        return raw_data


def _centre(img, BBOX):
    l = BBOX['max_r'] - BBOX['min_r']
    w = BBOX['max_c'] - BBOX['min_c']
    s = BBOX['max_s'] - BBOX['min_s']
    d = (l - w) // 2
    img_brain = img[BBOX['min_s']: BBOX['max_s'], BBOX['min_r']: BBOX['max_r'], BBOX['min_c']:BBOX['max_c']]
    img_brain_pad = np.zeros((s, l, l))
    img_brain_pad[:, :, d:w + d] = img_brain
    return img_brain_pad


def main(args):
    # reade the json file containing all the params
    with open(args.json, 'r') as f:
        cfg = json.loads(f.read())

    reader_params = cfg['reader']
    writer_params = cfg['writer']
    time_points = cfg['writer']['time_points']
    cv = writer_params['cross_validation']
    break_tfrecords = writer_params['break_tfrecords']

    reader = Reader(reader_params, writer_params['mode'], time_points)
    ids = reader.get_ids()

    nb_newt2_ids = {}

    if writer_params['debug']:
        ids = [ids[0][0:4]]

    if cv and writer_params['mode'] == 'train':
        for i in range(5):

            fold_ids = ids[i]

            ids_id = [fold_ids[i][0]+"_" + fold_ids[i][1] for i in range(len(fold_ids))]
            with open(join(writer_params['outdir'], "bravo_placebo_{}_{}_ids.txt".format(writer_params['mode'], i + 1)),
                      'w') as data_file:
                json.dump(ids_id, data_file, indent=4)

            writer = tf.python_io.TFRecordWriter(
                join(writer_params['outdir'], 'bravo_placebo_' + str(writer_params['mode'] + '_' + str(i + 1)) + '.tfrecords'))

            for _id in fold_ids:
                # read images and pre prcoess them
                input_, output_, nb_newt2, nb_gad = _pre_process(_id, writer_params, time_points)

                print("input:", np.shape(input_))
                print("output:", np.shape(output_))

                # create features
                feature = {'tp1': _bytes_feature(tf.compat.as_bytes(input_.tostring())),
                           'nb_newt2':_int64_feature(nb_newt2),
                           'nb_gad': _int64_feature(nb_gad),
                           'tp2': _bytes_feature(tf.compat.as_bytes(output_.tostring()))
                           }

                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

            writer.close()

    elif break_tfrecords:

        nb_folds = len(ids)

        print("nb_folds:", nb_folds)

        count = 0

        for i in range(nb_folds):

            ids_fold = ids[i]

            print("nb_subj_folds:", len(ids_fold))

            ids_id = [ids_fold[i][0] + "_" + ids_fold[i][1] for i in range(len(ids_fold))]
            with open(join(writer_params['outdir'], "bravo_placebo_{}_{}_ids.txt".format(writer_params['mode'], i+1)),
                      'w') as data_file:
                json.dump(ids_id, data_file, indent=4)

            writer = tf.python_io.TFRecordWriter(
                join(writer_params['outdir'], 'bravo_placebo_' + str(writer_params['mode'] + '_' + str(i + 1)) + '.tfrecords'))

            for _id in ids_fold:
                # read images and pre prcoess them

                input_, output_, nb_newt2, nb_gad = _pre_process(_id, writer_params, time_points)

                nb_newt2_ids['_'.join(_id[0:2])] = nb_newt2
                #print(count)
                count += 1

                # create features
                feature = {'tp1': _bytes_feature(tf.compat.as_bytes(input_.tostring())),
                           'nb_newt2': _int64_feature(nb_newt2),
                           'tp2': _bytes_feature(tf.compat.as_bytes(output_.tostring()))
                           }

                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

            writer.close()

    else:
        ids_id = [ids[i][0] + "_" + ids[i][1] for i in range(len(ids))]

        tot_nb_newt2 = []

        with open("/cim/nazsepah/data_tf/bravo_placebo_{}_ids.txt".format(writer_params['mode']),
                  'w') as data_file:
            json.dump(ids_id, data_file, indent=4)

        writer = tf.python_io.TFRecordWriter(
            join(writer_params['outdir'], 'bravo_placebo_' + str(writer_params['mode']) + '.tfrecords'))

        for _id in ids:
            # read images and pre prcoess them
            input_, output_, nb_newt2, nb_gad = _pre_process(_id, writer_params, time_points)
            tot_nb_newt2.append(nb_newt2)

            # create features
            feature = {'tp1': _bytes_feature(tf.compat.as_bytes(input_.tostring())),
                       'nb_newt2': _int64_feature(nb_newt2),
                       'tp2': _bytes_feature(tf.compat.as_bytes(output_.tostring()))
                       }

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()

    tot_nb_newt2 = [0, 0, 0, 0, 0]
    for _id in nb_newt2_ids.keys():
        nb_newt2 = nb_newt2_ids[_id]
        nb_newt2 = 4 if nb_newt2 >4 else nb_newt2
        tot_nb_newt2[nb_newt2] += 1

    print("total:", tot_nb_newt2)


def _get_cfg():
    parser = argparse.ArgumentParser(description="handler for writing the tfrecord flies",
                                     usage="python write_tfrecord.py -j configs/tfrecord.json")
    parser.add_argument("-j", "--json", help="configuration json file", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(_get_cfg())






