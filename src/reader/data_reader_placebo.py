import numpy as np
from os.path import join, exists, isdir
from os import listdir
from random import shuffle, seed

_MASK_PATH = {'MS-LAQ-302-STX': r'/cim/data/neurorx/MS-LAQ-302-STX/extra/anewt2_75/',
              '101MS326': r'/cim/data/neurorx/lesion_files/'}

class Reader:
    def __init__(self, config, mode, time_points):
        self.data_dir = config.get("image_dir", '/cim/data_raw/preproc/')
        self.dataset_tag = config.get("dataset_name", 'MS-LAQ-302-STX')
        self.inp_tag = config.get("input_tag", "icbm_N3_VP.mnc")
        self.mask_tag = config.get("mask_tag", "icbm_Beast.mnc")
        self.out_tag = config.get("output_tag", "icbm.mnc")
        self.time_points = time_points
        self.inp_mods = config.get("input_mods", {'t1p': 0, 'pdw': 1, 'flr': 2, 't2w': 3})
        self.out_mods = config.get("output_mods", {'ct2f': 0})
        self.seed = config.get("seed", 1333)
        self.break_tfrecords = config.get("break_tfrecords", False)
        self.mode = mode
        self.cv = config.get("cross_validation", True)
        self.nb_inp_mods = len(list(self.inp_mods.keys()))
        self.nb_out_mods = len(list(self.out_mods.keys()))
        self.ids = self._get_ids()

    def _get_ids(self):
        mri_paths = self._cal_ids()
        subj_ids = self._get_subject_ids(mri_paths)
        subj_ids = self._get_partitions(subj_ids)
        placebo_ids = self._get_placebo_ids(subj_ids)
        mri_paths = self._get_subjects_tps(mri_paths, subj_ids, placebo_ids)
        return mri_paths

    def _cal_ids(self):
        ids = {}
        path_info = [[data_dir, site, subj] for data_dir in self.data_dir for site in listdir(data_dir) for subj in
                     listdir(join(data_dir, site)) if '.scannerdb' not in subj]

        nb_subjs = 0
        for data_dir, site, subj in path_info:
            # set the subject_folder
            subject_folder = join(data_dir, site, subj)
            dataset_tag = self.dataset_tag

            for j in range(len(self.time_points) - 1):

                subject_tp_folder = join(subject_folder, self.time_points[j])
                next_subject_tp_folder = join(subject_folder, self.time_points[j+1])

                if all(isdir(path) for path in [subject_tp_folder, next_subject_tp_folder]):

                    # get all the paths
                    pths = ["" for _ in range(2 * self.nb_inp_mods + self.nb_out_mods)]# changed
                    for i, mod in enumerate(self.inp_mods):
                        img_name = '{}_{}_{}_{}_{}_{}'.format(dataset_tag, site, subj,
                                                              self.time_points[j],
                                                              mod, self.inp_tag)
                        next_img_name = '{}_{}_{}_{}_{}_{}'.format(dataset_tag, site, subj,
                                                                   self.time_points[j+1],
                                                                   mod, self.inp_tag)

                        pths[2 * i] = join(subject_tp_folder, img_name)
                        pths[2 * i + 1] = join(next_subject_tp_folder, next_img_name)

                    # get the newt2 file paths
                    if dataset_tag == 'MS-LAQ-302-STX':
                        newt2_img_name = '{}_{}_{}_{}_anewt2_75_TREF-{}.mnc.gz'.format(dataset_tag, site, subj,
                                                                                     self.time_points[j + 1],
                                                                                     self.time_points[j])
                        newt2_path = _MASK_PATH[dataset_tag]
                        pths[2 * len(list(self.inp_mods.keys()))] = join(newt2_path, newt2_img_name)

                    elif dataset_tag == '101MS326':

                        newt2_img_name = '{}_{}_{}_{}_newt2f_TREF-{}_ISPC-stx152lsq6.mnc.gz'.format(dataset_tag,
                                                                                                  site, subj,
                                                                                                  self.time_points[j + 1],
                                                                                                  self.time_points[j])

                        newt2_path = join(_MASK_PATH[dataset_tag], next_subject_tp_folder)
                        pths[2 * len(list(self.inp_mods.keys()))] = join(newt2_path, newt2_img_name)

                    else:
                        raise Exception('unknown dataset')

                    # check if all the paths exist, if so add the subject to the list of subjects
                    if np.all([exists(pth) for pth in pths]):
                        subj_dataset = subj
                        if subj_dataset not in ids.keys():
                            ids[subj_dataset] = {}
                            nb_subjs += 1
                        ids[subj_dataset][self.time_points[j]] = {'path': subject_folder,
                                                                  'dataset': dataset_tag,
                                                                  'site': site,
                                                                  'inp_tag': self.inp_tag,
                                                                  'mask_tag': self.mask_tag,
                                                                  'out_tag': self.out_tag}
        print("total number of subjects:", nb_subjs)
        return ids

    @staticmethod
    def _get_subject_ids(ids):
        return list(ids.keys())

    #@staticmethod
    def _get_subjects_tps(self, mri_paths, subj_ids, placebo_ids):
        sb_tp = []
        if self.cv or self.break_tfrecords:
            nb_folds = len(subj_ids)
            for i in range(nb_folds):
                sb_tp_fold = []
                subj_ids_fold = subj_ids[i]
                placebo_ids_fold = placebo_ids[i]
                ids_map = [_id.split('_')[-1] for _id in subj_ids_fold]
                for i, subject in enumerate(subj_ids_fold):
                    subj = subject.split('__')[-1]
                    tps = list(mri_paths[subject].keys())
                    subj_clinical_id = ids_map[i]
                    if subj_clinical_id in placebo_ids_fold:
                        for tp in tps:
                            info = [subj, tp, mri_paths[subject][tp]['path'], mri_paths[subject][tp]['dataset'],
                                    mri_paths[subject][tp]['site'], mri_paths[subject][tp]['inp_tag'],
                                    mri_paths[subject][tp]['mask_tag']]
                            sb_tp_fold.append(info)
                shuffle(sb_tp_fold)
                sb_tp.append(sb_tp_fold)
        else:
            ids_map = [_id.split('_')[-1] for _id in subj_ids]
            for i, subject in enumerate(subj_ids):
                subj = subject.split('__')[-1]
                tps = list(mri_paths[subject].keys())
                subj_clinical_id = ids_map[i]
                if subj_clinical_id in placebo_ids:
                    for tp in tps:
                            info = [subj, tp, mri_paths[subject][tp]['path'], mri_paths[subject][tp]['dataset'],
                                    mri_paths[subject][tp]['site'], mri_paths[subject][tp]['inp_tag'],
                                    mri_paths[subject][tp]['mask_tag']]
                            sb_tp.append(info)
            shuffle(sb_tp)
        return sb_tp

    def _get_partitions(self, subj_ids):
        mode = self.mode
        seed_val = self.seed
        seed(seed_val)
        subj_ids_s = sorted(subj_ids)
        shuffle(subj_ids_s)
        nb_subjects = len(subj_ids_s)
        if self.cv:
            train_split = int(nb_subjects * 0.9)
            if mode == 'test':
                partitions = subj_ids_s[train_split:]
            elif mode == 'train':
                subj_ids_train = subj_ids_s[:train_split]
                partitions = []
                fold_size = train_split // 5
                folds_ids = [0, fold_size, fold_size * 2, fold_size * 3, fold_size * 4, train_split]
                for i in range(5):
                    start_id = folds_ids[i]
                    end_id = folds_ids[i + 1]
                    fold_ids = subj_ids_train[start_id:end_id]
                    partitions.append(fold_ids)
        else:
            if self.break_tfrecords:
                train_split = int(nb_subjects * 0.6)
                val_split = int(nb_subjects * 0.8)
                partitions = []
                if mode == 'train':
                    subj_ids_train = subj_ids_s[:train_split]
                    fold_size = train_split // 6
                    folds_ids = [0, fold_size, fold_size * 2, fold_size * 3, fold_size * 4, fold_size * 5, train_split]
                    for i in range(6):
                        start_id = folds_ids[i]
                        end_id = folds_ids[i + 1]
                        fold_ids = subj_ids_train[start_id:end_id]
                        partitions.append(fold_ids)
                elif mode == 'valid':
                    partitions.append(subj_ids_s[train_split:val_split])
                elif mode == 'test':
                    partitions.append(subj_ids_s[val_split:])

            else:
                train_split = int(nb_subjects * 0.8)
                val_split = int(nb_subjects * 0.9)
                if mode == 'train':
                    partitions = subj_ids_s[:train_split]
                elif mode == 'valid':
                    partitions = subj_ids_s[train_split:val_split]
                elif mode == 'test':
                    partitions = subj_ids_s[val_split:]
        return partitions

    def _get_placebo_ids(self, ids):
        import csv
        id_attr = 'SUBJECT_Screening_Number'
        drug_attr = 'SUBJECT_Trial_Arm'
        if self.dataset_tag == 'MS-LAQ-302-STX':
            clinical_file = r'/cim/data/clinical-data/ipmsa.BRAVO.CLIN.20191007-144156.csv'
        elif self.dataset_tag == '101MS326':
            clinical_file = r'/cim/data/clinical-data/ipmsa.ASCEND.CLIN.20191007-134538.csv'
        placebo_ids = []
        if self.cv or self.break_tfrecords:
            nb_folds = len(ids)
            for i in range(nb_folds):
                placebo_ids_fold = []
                ids_map = [_id.split('_')[-1] for _id in ids[i]]
                csvreader = csv.reader(open(clinical_file, 'r'))
                csvheader = next(csvreader)
                id_indx = csvheader.index(id_attr)
                drug_indx = csvheader.index(drug_attr)
                for i, row in enumerate(csvreader):
                    try:
                        subj = row[id_indx]
                        drug = row[drug_indx].strip()
                        if (subj in ids_map) and (drug == 'Placebo'):
                            if subj not in list(placebo_ids_fold):
                                placebo_ids_fold.append(subj)
                    except:
                        continue
                #print(len(placebo_ids_fold))
                placebo_ids.append(placebo_ids_fold)

        else:
            ids_map = [_id.split('_')[-1] for _id in ids]
            csvreader = csv.reader(open(clinical_file, 'r'))
            csvheader = next(csvreader)
            id_indx = csvheader.index(id_attr)
            drug_indx = csvheader.index(drug_attr)
            for i, row in enumerate(csvreader):
                try:
                    subj = row[id_indx]
                    drug = row[drug_indx]
                    if (subj in ids_map) and (drug == 'Placebo'):
                        if subj not in list(placebo_ids):
                            placebo_ids.append(subj)
                except:
                    continue
        return placebo_ids

    def get_ids(self):
        return self.ids

    def get_nb_samples(self):
        return len(self.ids)

    def get_input_modalities(self):
        return self.inp_mods

    def get_output_modalities(self):
        return self.out_mods
