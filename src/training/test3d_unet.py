import os, sys, json, argparse
from os.path import join

from src.data_source.volume_generator_tfrecord import BrainDataProvider as tfDataProvider
from src.callbacks.callbacks_cnn import PlotCM, PlotTSNE, PlotCMProb, DISC, PlotROC, ACC


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
    outdir = expt_cfg['outdir']
    tfdir = expt_cfg['tfdir']

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(join(outdir, expt_name), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'recons'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'rocs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'counts'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'covs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'zs'), exist_ok=True)
    os.makedirs(join(outdir, expt_name, 'log'), exist_ok=True)

    gen_callback = tfDataProvider(tfdir, cfg['callback'])
    callback_data = gen_callback.data_generator()
    nb_samples_callback = gen_callback.get_nb_samples()

    cm = PlotCM(cfg, reuse=False)
    cm.on_epoch_end(callback_data, nb_imgs=nb_samples_callback, epoch=expt_cfg['epoch'])

    #acc = ACC(cfg, reuse=True)
    #acc.on_epoch_end(callback_data, nb_imgs=nb_samples_callback, mode_value=False, epoch=expt_cfg['epoch'])

    roc = PlotROC(cfg, reuse=True)
    roc.on_epoch_end(callback_data, nb_imgs=nb_samples_callback, epoch=expt_cfg['epoch'])

    #disc = DISC(cfg, reuse=True)
    #disc.on_epoch_end(callback_data, nb_imgs=nb_samples_callback, epoch=expt_cfg['epoch'])

    #cm_prob = PlotCMProb(cfg, callback_data, reuse=True)
    #cm_prob.on_epoch_end(nb_imgs=nb_samples_callback, nb_samples=10, epoch=expt_cfg['epoch'])

    #tsne = PlotTSNE(cfg, callback_data, reuse=True)
    #tsne.on_epoch_end(nb_imgs=nb_samples_callback, epoch=expt_cfg['epoch'])


if __name__ == "__main__":
    _main(_get_cfg())

