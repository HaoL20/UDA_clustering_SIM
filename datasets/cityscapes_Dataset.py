# -*- coding: utf-8 -*-
import random
from PIL import Image, ImageOps, ImageFilter, ImageFile
import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as ttransforms
import imageio
import time
import glob
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEBUG = False

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
NUM_CLASSES = 19

# colour map
label_colours_19 = [
    # [  0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]]  # the color of ignored label(-1)
label_colours_19 = list(map(tuple, label_colours_19))

# [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
# colour map
label_colours_16 = [
    # [  0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 60, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]]  # the color of ignored label(-1)
label_colours_16 = list(map(tuple, label_colours_16))


class City_Dataset(data.Dataset):
    def __init__(self,
                 args,
                 data_root_path=os.path.abspath('./datasets/Cityscapes'),
                 list_path=os.path.abspath('./datasets/Cityscapes'),
                 split='train',
                 base_size=769,
                 crop_size=769,
                 training=True,
                 class_16=False,
                 class_13=False,
                 load_pseudo=False,
                 pseudo_floder='',
                 n_pseudo_floder='',
                 pseudo_round=0):

        self.args = args
        self.data_path = data_root_path
        self.list_path = list_path
        self.split = split
        if DEBUG: print('DEBUG: Cityscapes {0:} dataset path is {1:}'.format(self.split, self.list_path))
        self.base_size = base_size
        self.crop_size = crop_size
        if DEBUG: print('DEBUG: Cityscapes {0:} dataset image size is {1:}'.format(self.split, self.crop_size))

        self.base_size = self.base_size if isinstance(self.base_size, tuple) else (self.base_size, self.base_size)
        self.crop_size = self.crop_size if isinstance(self.crop_size, tuple) else (self.crop_size, self.crop_size)
        self.training = training

        self.random_mirror = args.random_mirror
        self.resize = args.resize
        self.gaussian_blur = args.gaussian_blur
        self.color_jitter = args.color_jitter

        self.load_pseudo = load_pseudo
        self.pseudo_floder = pseudo_floder
        self.n_pseudo_floder = n_pseudo_floder
        self.pseudo_round = pseudo_round

        ###
        item_list_filepath = os.path.join(self.list_path, self.split + ".txt")
        try:
            time.sleep(2)
            self.items = [id for id in open(item_list_filepath)]
        except FileNotFoundError:
            print('sys.argv', os.path.dirname(sys.argv[0]))
            print('FileNotFoundError: cwdir is ' + os.getcwd())
            print(os.listdir(os.getcwd()))
            print('FileNotFoundError: parent cwdir is ' + os.path.dirname(os.getcwd()))
            print('glob:', glob.glob(os.path.dirname(os.getcwd()) + '/*'))
            print('os:', os.listdir(os.path.dirname(os.getcwd())))
            print('FileNotFoundError: parent of parent cwdir is ' + os.path.dirname(os.path.dirname(os.getcwd())))
            print(os.listdir(os.path.dirname(os.path.dirname(os.getcwd()))))
            self.items = [id for id in open(item_list_filepath)]
        ###

        ignore_label = -1

        ###
        # self.id_to_trainid = {i:i for i in range(-1,19)}
        # self.id_to_trainid[255] = ignore_label

        self.id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                              3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                              7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                              14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                              18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
        ###

        # In SYNTHIA-to-Cityscapes case, only consider 16 shared classes
        self.class_16 = class_16
        ###
        synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_16id = {id: i for i, id in enumerate(synthia_set_16)}
        self.trainid_to_16id[255] = ignore_label
        ###

        self.class_13 = class_13
        synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
        self.trainid_to_13id = {id: i for i, id in enumerate(synthia_set_13)}

        if DEBUG: print('DEBUG: Cityscapes {0:} -> item_list_filepath: {1:} , first item: {2:}'.format(self.split, item_list_filepath, self.items[0]))
        if DEBUG: print("{} num images in Cityscapes {} set have been loaded.".format(len(self.items), self.split))
        if self.args.numpy_transform:
            if DEBUG: print("used numpy_transform, instead of tensor transform!")

    def id2trainId(self, label, reverse=False, ignore_label=-1):
        if self.load_pseudo:
            label[label == 255] = ignore_label
            return label

        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        if self.class_16:
            label_copy_16 = ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_16id.items():
                label_copy_16[label_copy == k] = v
            label_copy = label_copy_16
        if self.class_13:
            label_copy_13 = ignore_label * np.ones(label.shape, dtype=np.float32)
            for k, v in self.trainid_to_13id.items():
                label_copy_13[label_copy == k] = v
            label_copy = label_copy_13
        return label_copy

    def __getitem__(self, item):
        ###
        id_img, id_gt = self.items[item].strip('\n').split(' ')
        image_path = self.data_path + id_img
        image = Image.open(image_path).convert("RGB")
        if item == 0 and DEBUG: print('DEBUG: Cityscapes {0:} -> image_path: {1:}'.format(self.split, image_path))
        ###

        if self.load_pseudo:  # Use pseudo-labels only during training
            assert self.split == 'train'
            id_plabel_path = self.pseudo_floder + id_gt
            id_nlabel_path = self.n_pseudo_floder + id_gt
            mask = []
            if os.path.exists(id_plabel_path):
                mask.append(Image.open(id_plabel_path))
            if os.path.exists(id_nlabel_path):
                mask.append(Image.open(id_nlabel_path))
            image, mask = self._pseudo_train_sync_transform(image, mask)
            # return image, mask[0], mask[1], id_gt
            return image, mask[0], id_gt  # 只返回获取标签

        ###
        gt_image_path = self.data_path + id_gt

        # gt_image = imageio.imread(gt_image_path,format='PNG-FI')[:,:,0]
        # gt_image = Image.fromarray(np.uint8(gt_image))

        gt_image = Image.open(gt_image_path)

        if item == 0 and DEBUG: print('DEBUG: Cityscapes {0:} -> gt_path: {1:}'.format(self.split, gt_image_path))
        ###

        if (self.split == "train" or self.split == "trainval") and self.training:
            image, gt_image = self._train_sync_transform(image, gt_image)
        else:
            image, gt_image = self._val_sync_transform(image, gt_image)

        return image, gt_image, id_gt

    def _train_sync_transform(self, img, mask):
        if self.random_mirror:  # default = True
            # random mirror
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if mask: mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            crop_w, crop_h = self.crop_size

        if self.resize:  # default = True
            img = img.resize(self.crop_size, Image.BICUBIC)
            if mask: mask = mask.resize(self.crop_size, Image.NEAREST)

        if self.args.gaussian_blur:  # default = True, when gen pseudo,set to False
            # gaussian blur as in PSP
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 1.15)))
        # final transform
        if mask:
            img, mask = self._img_transform(img), self._mask_transform(mask)
            return img, mask
        else:
            img = self._img_transform(img)
            return img

    def _val_sync_transform(self, img, mask, resize_mask=True):

        if self.resize:
            img = img.resize(self.crop_size, Image.BICUBIC)
            if resize_mask:
                mask = mask.resize(self.crop_size, Image.NEAREST)

        # final transform
        img, mask = self._img_transform(img, val=True), self._mask_transform(mask)
        return img, mask

    def _pseudo_train_sync_transform(self, img, mask):

        n_mask = len(mask)
        if self.random_mirror:  # default = True
            # random mirror
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                for i in range(n_mask):
                    mask[i] = mask[i].transpose(Image.FLIP_LEFT_RIGHT)

        if self.resize:  # default = True
            img = img.resize(self.crop_size, Image.BICUBIC)
            for i in range(n_mask):
                mask[i] = mask[i].resize(self.crop_size, Image.NEAREST)

        if self.gaussian_blur:  # default = True
            # gaussian blur as in PSP
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 1.15)))

        # final transform
        img = self._img_transform(img)
        for i in range(n_mask):
            mask[i] = self._mask_transform(mask[i])
        return img, mask

    def _img_transform(self, image, val=False):
        if self.args.numpy_transform == True and self.color_jitter == True:
            assert False,"can`t color_jitter with numpy_transform"
        if self.args.numpy_transform:  # default = False
            image = np.asarray(image, np.float32)
            image = image[:, :, ::-1]  # change to BGR
            image -= IMG_MEAN
            image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
            new_image = torch.from_numpy(image)
        else:
            transforms_list = [
                ttransforms.ToTensor(),
                ttransforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            ]
            if not val and self.color_jitter and random.random() > 0.5:     # not color_jitter when val
                transforms_list.insert(0, ttransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2))
            image_transforms = ttransforms.Compose(transforms_list)
            new_image = image_transforms(image)
        return new_image

    def _mask_transform(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target

    def __len__(self):
        return len(self.items)


class City_DataLoader():
    def __init__(self, args, training=True):

        self.args = args

        data_set = City_Dataset(args,
                                data_root_path=self.args.data_root_path,
                                list_path=self.args.list_path,
                                split=args.split,
                                base_size=args.base_size,
                                crop_size=args.crop_size,
                                training=training,
                                class_16=args.class_16,
                                class_13=args.class_13)

        if (self.args.split == "train" or self.args.split == "trainval") and training:
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)
        else:
            self.data_loader = data.DataLoader(data_set,
                                               batch_size=self.args.batch_size,
                                               shuffle=False,
                                               num_workers=self.args.data_loader_workers,
                                               pin_memory=self.args.pin_memory,
                                               drop_last=True)

        val_set = City_Dataset(args,
                               data_root_path=self.args.data_root_path,
                               list_path=self.args.list_path,
                               split='test' if self.args.split == "test" else 'val',
                               base_size=args.base_size,
                               crop_size=args.crop_size,
                               training=False,
                               class_16=args.class_16,
                               class_13=args.class_13)
        self.val_loader = data.DataLoader(val_set,
                                          batch_size=self.args.batch_size,
                                          shuffle=False,
                                          num_workers=self.args.data_loader_workers,
                                          pin_memory=self.args.pin_memory,
                                          drop_last=True)
        self.valid_iterations = (len(val_set) + self.args.batch_size) // self.args.batch_size

        self.num_iterations = (len(data_set) + self.args.batch_size) // self.args.batch_size


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(torch.arange(x.size(i) - 1, -1, -1).tolist()).long()  # 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。
                 for i in range(x.dim()))
    return x[inds]


def inv_preprocess(imgs, numages=1, img_mean=IMG_MEAN, numpy_transform=False):
    """Inverse preprocessing of the batch of images.
       
    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
      numpy_transform: whether change RGB to BGR during img_transform.
  
    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    if numpy_transform:
        imgs = flip(imgs, 1)  # change to RGB

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)

    norm_ip(imgs, float(imgs.min()), float(imgs.max()))
    return imgs


def decode_labels(mask, num_classes, num_images=1):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """

    assert num_classes == 16 or num_classes == 19
    label_colours = label_colours_16 if num_classes == 16 else label_colours_19

    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)


def colorize_mask(mask, num_classes):
    assert num_classes == 16 or num_classes == 19
    label_colours = label_colours_16 if num_classes == 16 else label_colours_19
    palettes = []
    for label_colour in label_colours:
        palettes = palettes + list(label_colour)
    palettes = palettes + [255, 255, 255] * (256 - len(palettes))

    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palettes)
    return new_mask


name_classes = [
    'road',
    'sidewalk',
    'building',
    'wall',
    'fence',
    'pole',
    'trafflight',
    'traffsign',
    'vegetation',
    'terrain',
    'sky',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle',
    'unlabeled'
]
