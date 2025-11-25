import logging
import math
import random

import PIL.ImageChops
import cv2
import torch
import torch.utils.data as data
import json
import PIL.Image
import os
import os.path
import pandas as pd
import numpy as np
from .manipulation import *


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions, mode='train'):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    if mode == 'train':
                        if os.path.exists(os.path.dirname(path).replace('FF++_Faces_c40', 'heat_map_fusion')):
                            # print(os.path.dirname(path).replace('FF++_Faces_c40', 'FF++_faces_new'))
                            item = (path, class_to_idx[target])
                            images.append(item)
                    else:
                        item = (path, class_to_idx[target])
                        images.append(item)
                    # else:
                    #     print(path.replace('FF++_Faces_c40', 'FF++_faces_new'))
    return images


def make_ffpp_dataset_subset(dir, class_to_idx, extensions, subsets, mask_ratio=None):
    """
    path/train/fake/Deepfakes/video
    """
    images = []
    dir = os.path.expanduser(dir)
    logging.info("%s stage: use the subset %s" % (os.path.basename(dir), str(subsets)))
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        if target == 'real':
            if mask_ratio is None:
                d_sub = os.path.join(d, 'youtube')
            else:
                d_sub = os.path.join(d, 'youtube', 'per_%s' % str(mask_ratio))
            for root, _, fnames in sorted(os.walk(d_sub)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

        elif target == 'fake':
            for subset_name in subsets:
                if mask_ratio is None:
                    d_sub = os.path.join(d, subset_name)
                else:
                    d_sub = os.path.join(d, subset_name, 'per_%s' % str(mask_ratio))
                for root, _, fnames in sorted(os.walk(d_sub)):
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, extensions):
                            path = os.path.join(root, fname)
                            item = (path, class_to_idx[target])
                            images.append(item)
    return images


def make_dataset_race(csv_list, class_to_idx, bio_choose='full', phase='train'):
    import csv
    images = []
    images_bio_choose = {
            'male': {
                '1': [],
                '2': [],
                '3': [],
                '4': [],
                '5': [],
                '6': [],
                '7': [],
                '8': [],
                '9': [],
                '10': [],
            },
            'female': {
                '1': [],
                '2': [],
                '3': [],
                '4': [],
                '5': [],
                '6': [],
                '7': [],
                '8': [],
                '9': [],
                '10': [],
            },
        }
    save_dict = {}
    cnt_real = 0
    cnt_fake = 0
    # img_path, label, ismale, isasian, iswhite, isblack, intersec_label, spe_label
    for csv_file in csv_list:
        with open(csv_file, newline='', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                row = [x.strip() for x in row]
                ori_path = row[0]
                if phase == 'train':
                    img_path = os.path.join('/home/harryxa', row[0][1:])
                    img_path = img_path.replace('GANs', 'GAN'
                                                ).replace('DMs', 'DMS'
                                                          ).replace('StyleGAN2', 'StyleGAN2_FFHQ'
                                                                    ).replace('stylegan3', 'stylegan3_T_FFHQU_processed'
                                                                              ).replace('ProGAN images', 'ProGAN_images'
                                                                                        ).replace('latent_diffusion',
                                                                                                  'latent_diffusion_FFHQ'
                                                                                                  ).replace(
                        'taming_transformer_VQGAN', 'taming_transformer:VQGAN'
                        ).replace('Palette', 'Palette_CelebAHQ'
                                  ).replace('StableDiffusion1.5', 'StableDiffusion1.5_wiki_processed_process'
                                            ).replace('StableDiffusion_Inpainting',
                                                      'StableDiffusion_Inpainting_wiki_processed_process'
                                                      ).replace('MMD_GAN', 'MMD_GAN_CelebA'
                                                                ).replace('STGAN', 'STGAN_CelebA'
                                                                          ).replace('STGAN_CelebA_', 'STGAN_').replace('CommercialTools',
                                                                                    'CommercialTools_CelebAHQ_processed')
                    save_dict[img_path.split('/')[3] + '_' + img_path.split('/')[4] + '_' + img_path.split('/')[5]] = 1
                    img_key = img_path.replace('AI_Face_imagesV2', 'AIFace')
                else:
                    img_path = os.path.join('/home/harryxa/AI_Face_imagesV2/test', row[0][1:])

                is_male = row[1]
                # print(row)

                if phase == 'train':
                    label = row[4]
                    if 'real' in img_path or 'Real' in img_path:
                        label = 0
                    if label == 0:
                        cnt_real += 1
                    else:
                        cnt_fake += 1
                    skin_tone = row[3]
                    intersection = int(row[6])
                    spe_label = 0
                    if label == 1:
                        spe_label = int(row[6])
                else:
                    # print(row)
                    label = row[2]
                    if 'real' in img_path or 'Real' in img_path:
                        label = 0
                    intersection = int(row[4])
                    skin_tone = row[1]
                    spe_label = 0

                gender = 'male' if int(is_male) == 1 else 'female'
                class_label = int(label)
                race = skin_tone

                if bio_choose != 'full':
                    bio_choose_gender, bio_choose_race = bio_choose.split('_')
                    if gender == bio_choose_gender and race == bio_choose_race:
                        images.append((img_path, label, gender, race, intersection, ori_path))
                else:
                    images.append((img_path, class_label, gender, race, intersection, ori_path))

                if images_bio_choose.get(gender).get(race) is None:
                    images_bio_choose[gender][race] = []
                images_bio_choose[gender][race].append((img_path, label, gender, race, intersection))

    print('bio_choose: ', bio_choose)
    print('len of dataset:', len(images))
    return images, images_bio_choose


class FrameRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])




class DatasetFolder_race(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, csv_list, mask_path=None, transform=None, target_transform=None,
                 root_2=None,
                 config=None, bio_choose='full', phase='train'):
        classes = {'real', 'fake'}
        class_to_idx = {'1': 0, '0': 1}

        self.root = root
        self.dataset_mode = os.path.basename(self.root)

        samples, images_bio_choose = make_dataset_race(csv_list,
                                                       class_to_idx,
                                                       bio_choose,
                                                       phase=phase)

        self.bio_acc = {
            'male': {
                '1': 0,
                '2': 0,
                '3': 0,
                '4': 0,
                '5': 0,
                '6': 0,
                '7': 0,
                '8': 0,
                '9': 0,
                '10': 0,
            },
            'female': {
                '1': 0,
                '2': 0,
                '3': 0,
                '4': 0,
                '5': 0,
                '6': 0,
                '7': 0,
                '8': 0,
                '9': 0,
                '10': 0,
            },
        }

        self.bio_ratio = {
            'male': {
                '1': 0,
                '2': 0,
                '3': 0,
                '4': 0,
                '5': 0,
                '6': 0,
                '7': 0,
                '8': 0,
                '9': 0,
                '10': 0,
            },
            'female': {
                '1': 0,
                '2': 0,
                '3': 0,
                '4': 0,
                '5': 0,
                '6': 0,
                '7': 0,
                '8': 0,
                '9': 0,
                '10': 0,
            },
        }

        print('dataset samples:', len(samples))
        if phase == 'train':
            for gender in images_bio_choose:
                for race in images_bio_choose[gender]:
                    print('Dataset of bio choose {gender}-{race}: {lenth}'.format(gender=gender, race=race, lenth=len(images_bio_choose[gender][race])))
                    self.bio_acc[gender][race] += len(images_bio_choose[gender][race])
        else:
            for gender in images_bio_choose:
                for race in images_bio_choose[gender]:
                        print('Dataset of bio choose {gender}-{race}: {lenth}'.format(gender=gender, race=race,lenth=len(images_bio_choose[gender][race])))
                        if images_bio_choose[gender][race] is not None:
                            self.bio_acc[gender][race] += len(images_bio_choose[gender][race])

        for gender in images_bio_choose:
            for race in images_bio_choose[gender]:
                self.bio_ratio[gender][race] = self.bio_acc[gender][race]/len(samples)
        # print('dataset images_bio_choose:', bio_choose)

        print(self.bio_ratio)
        self.bio_images = images_bio_choose

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                extensions)))

        self.loader = loader
        self.extensions = extensions

        self.config = config

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        self.bio_choose = bio_choose
        self.phase = phase

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.phase == 'train':
            path, target, gender, race, spe_label, ori_path = self.samples[index]
            sample = self.loader(path)
            # print(gender)

            waiting_list = []
            male_list = ['male', 'female']
            race_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

            male_race_list = [
                'male_1',
                'male_2',
                'male_3',
                'male_4',
                'male_5',
                'male_6',
                'male_7',
                'male_8',
                'male_9',
                'male_10',
                'female_1',
                'female_2',
                'female_3',
                'female_4',
                'female_5',
                'female_6',
                'female_7',
                'female_8',
                'female_9',
                'female_10',

            ]

            for item in male_race_list:
                wait_gender, wait_race = item.split('_')
                if wait_gender != gender and wait_race != race:
                    if race in ['1', '2', '3'] and wait_race in ['1', '2', '3']:
                        continue
                    elif race in ['4', '5', '6'] and wait_race in ['4', '5', '6']:
                        continue
                    elif race in ['7', '8', '9', '10'] and wait_race in ['7', '8', '9', '10']:
                        continue
                    else:
                        waiting_list.append(item)

            bio_contras_choose = random.choice(waiting_list)
            contras_gender, contras_race = bio_contras_choose.split('_')
            contras_race = str(contras_race)
            # if int(spe_label) <= 3:
            #     contras_spe_label = int(spe_label) + 3
            # else:
            #     contras_spe_label = int(spe_label) - 3

            # print(contras_gender, contras_race)
            # print(len(self.bio_images[contras_gender][contras_race]))
            new_ind = random.randint(0, len(self.bio_images[contras_gender][contras_race]) - 1)
            # print(self.bio_images[contras_gender][contras_race][new_ind])
            contras_sample, contras_target, contras_gender, contras_race, contras_spe = \
            self.bio_images[contras_gender][contras_race][new_ind]
            contras_sample = self.loader(contras_sample)

            # if gender == 'male':
            #     new_ind = random.randint(0, len(self.bio_acc[gender][race]) - 1)
            #     contras_sample, contras_target, contras_gender, contras_race, contras_spe = self.images_female[spe_label][new_ind]
            #     contras_sample = self.loader(contras_sample)
            # else:
            #     new_ind = random.randint(0, len(self.images_male[spe_label]) - 1)
            #     contras_sample, contras_target, contras_gender, contras_race, contras_spe = self.images_male[spe_label][new_ind]
            #     contras_sample = self.loader(contras_sample)

            # print(contras_target, contras_gender, contras_race)
            if self.transform is not None:
                sample = self.transform(sample)
                contras_sample = self.transform(contras_sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target, gender, race, contras_sample, self.bio_ratio[gender][race]
        else:
            path, target, gender, race, spe_label, ori_path = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target, gender, race, path, ori_path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        try:
            img = PIL.Image.open(f)
        except:
            print(path)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class ImageFolder_race(DatasetFolder_race):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, csv_list, mask_path=None, transform=None, target_transform=None, config=None,
                 loader=default_loader, bio_choose='full', phase='train'):
        super(ImageFolder_race, self).__init__(root, loader, IMG_EXTENSIONS,
                                               csv_list=csv_list,
                                               mask_path=mask_path,
                                               transform=transform,
                                               config=config,
                                               target_transform=target_transform,
                                               bio_choose=bio_choose,
                                               phase=phase)
        self.imgs = self.samples
