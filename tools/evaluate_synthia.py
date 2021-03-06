import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from math import ceil
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
from PIL import Image

import sys
sys.path.append(os.path.abspath('.'))
from datasets.cityscapes_Dataset import inv_preprocess, decode_labels, colorize_mask
from datasets.gta5_Dataset import GTA5_DataLoader
from datasets.synthia_Dataset import SYNTHIA_DataLoader
from tools.train_source import Eval, add_train_args, str2bool, init_args
from utils.train_helper import get_model


class Evaluater():
    def __init__(self, args, cuda=None, train_id=None, logger=None):
        self.args = args
        self.cuda = cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        self.current_MIoU = 0
        self.best_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.train_id = train_id
        self.logger = logger

        # set TensorboardX
        self.writer = SummaryWriter(self.args.checkpoint_dir)

        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        # loss definition
        self.loss = nn.CrossEntropyLoss(ignore_index= -1)
        self.loss.to(self.device)

        # model
        self.model, params = get_model(self.args)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        self.model.to(self.device)
        self.original_interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            path1 = os.path.join(*self.args.checkpoint_dir.split('/')[:-1], self.train_id + 'best.pth')
            path2 = self.args.pretrained_ckpt_file
            if os.path.exists(path1):
                pretrained_ckpt_file = path1
            elif os.path.exists(path2):
                pretrained_ckpt_file = path2
            else:
                raise AssertionError("no pretrained_ckpt_file")
            self.load_checkpoint(pretrained_ckpt_file)

        # dataloader
        self.dataloader = SYNTHIA_DataLoader(self.args) if self.args.dataset=="synthia" else GTA5_DataLoader(self.args)
        self.dataloader.val_loader = self.dataloader.data_loader
        self.dataloader.valid_iterations = min(self.dataloader.num_iterations, 500)
        self.epoch_num = ceil(self.args.iter_max / self.dataloader.num_iterations)

    def main(self):

        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
            self.logger.info("{:16} {}".format(key, val))

        # choose cuda
        if self.cuda:
            current_device = torch.cuda.current_device()
            self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))
        else:
            self.logger.info("This model will run on CPU")

        # validate
        self.validate()

        self.writer.close()

    def validate(self):
        self.logger.info('validating one epoch...')
        self.Eval.reset()
        with torch.no_grad():
            tqdm_batch = tqdm(self.dataloader.val_loader, total=self.dataloader.valid_iterations, desc="Val Epoch-{}-".format(self.current_epoch + 1))
            self.model.eval()
            i = 0

            for x, y, id in tqdm_batch:
                i += 1
                if self.cuda:
                    x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)

                # model
                pred = self.model(x)[0]
                if args.get_original_label:
                    pred = self.original_interp(pred)
                if args.save_prediction_to is not None:
                    save_pred = self.original_interp(pred).cpu().data[0].numpy()
                    save_pred = save_pred.transpose(1, 2, 0)
                    save_pred = np.asarray(np.argmax(save_pred, axis=2), dtype=np.uint8)
                    output_col = colorize_mask(save_pred, self.args.num_classes)
                    output = Image.fromarray(save_pred)

                    id_gt = id_gt[0]
                    id_mask_dir = os.path.dirname(os.path.join(args.save_prediction_to, 'id_mask', id))
                    if not os.path.exists(id_mask_dir):
                        os.makedirs(id_mask_dir)
                    color_mask_dir = os.path.dirname(os.path.join(args.save_prediction_to, 'color_mask', id))
                    if not os.path.exists(color_mask_dir):
                        os.makedirs(color_mask_dir)
                    output.save(os.path.join(args.save_prediction_to, 'id_mask', id))
                    output_col.save(os.path.join(args.save_prediction_to, 'color_mask', id))

                y = torch.squeeze(y, 1)

                if self.args.flip:
                    pred_P = F.softmax(pred, dim=1)
                    def flip(x, dim):
                        dim = x.dim() + dim if dim < 0 else dim
                        inds = tuple(slice(None, None) if i != dim
                                else x.new(torch.arange(x.size(i)-1, -1, -1).tolist()).long()
                                for i in range(x.dim()))
                        return x[inds]
                    x_flip = flip(x, -1)
                    pred_flip = self.model(x_flip)[0]
                    pred_P_flip = F.softmax(pred_flip, dim=1)
                    pred_P_2 = flip(pred_P_flip, -1)
                    pred_c = (pred_P+pred_P_2)/2
                    pred = pred_c.data.cpu().numpy()
                else:
                    pred = pred.data.cpu().numpy()
                label = y.cpu().numpy()

                argpred = np.argmax(pred, axis=1)


                self.Eval.add_batch(label, argpred)

                if i == self.dataloader.valid_iterations:
                    break
                
                if i % 20 ==0 and self.args.image_summary:
                    #show val result on tensorboard
                    images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
                    labels_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)
                    preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
                    for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                        self.writer.add_image('eval/'+ str(index)+'/Images', img, self.current_epoch)
                        self.writer.add_image('eval/'+ str(index)+'/Labels', lab, self.current_epoch)
                        self.writer.add_image('eval/'+ str(index)+'/preds', color_pred, self.current_epoch)
            #show val result on tensorboard
            if self.args.image_summary:
                images_inv = inv_preprocess(x.clone().cpu(), self.args.show_num_images, numpy_transform=self.args.numpy_transform)
                labels_colors = decode_labels(label, self.args.num_classes, self.args.show_num_images)
                preds_colors = decode_labels(argpred, self.args.num_classes, self.args.show_num_images)
                for index, (img, lab, color_pred) in enumerate(zip(images_inv, labels_colors, preds_colors)):
                    self.writer.add_image('0Images/'+str(index), img, self.current_epoch)
                    self.writer.add_image('a'+str(index)+'/Labels', lab, self.current_epoch)
                    self.writer.add_image('a'+str(index)+'/preds', color_pred, self.current_epoch)

            # get eval result
            if self.args.class_16:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA_16, MPA_13 = Eval.Mean_Pixel_Accuracy()
                    MIoU_16, MIoU_13 = Eval.Mean_Intersection_over_Union()
                    FWIoU_16, FWIoU_13 = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC_16, PC_13 = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_16:{:.3f}, MIoU_16:{:.3f}, FWIoU_16:{:.3f}, PC_16:{:.3f}'.format(self.current_epoch, name, PA, MPA_16, MIoU_16, FWIoU_16, PC_16))
                    self.logger.info('\nEpoch:{:.3f}, {} PA:{:.3f}, MPA_13:{:.3f}, MIoU_13:{:.3f}, FWIoU_13:{:.3f}, PC_13:{:.3f}'.format(self.current_epoch, name, PA, MPA_13, MIoU_13, FWIoU_13, PC_13))
                    return PA, MPA_16, MIoU_16, FWIoU_16
            else:
                def val_info(Eval, name):
                    PA = Eval.Pixel_Accuracy()
                    MPA = Eval.Mean_Pixel_Accuracy()
                    MIoU = Eval.Mean_Intersection_over_Union()
                    FWIoU = Eval.Frequency_Weighted_Intersection_over_Union()
                    PC = Eval.Mean_Precision()
                    print("########## Eval{} ############".format(name))

                    self.logger.info('\nEpoch:{:.3f}, {} PA1:{:.3f}, MPA1:{:.3f}, MIoU1:{:.3f}, FWIoU1:{:.3f}, PC:{:.3f}'.format(self.current_epoch, name, PA, MPA, MIoU, FWIoU, PC))
                    return PA, MPA, MIoU, FWIoU

            PA, MPA, MIoU, FWIoU = val_info(self.Eval, "")
                
            self.Eval.Print_Every_class_Eval(logger = self.logger)
            tqdm_batch.close()

        return PA, MPA, MIoU, FWIoU


    def load_checkpoint(self, filename):
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename, map_location=torch.device(self.device))

            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.module.load_state_dict(checkpoint)
            self.logger.info("Checkpoint loaded successfully from "+filename)

            if 'crop_size' in checkpoint:
                self.args.crop_size = checkpoint['crop_size']
                print(checkpoint['crop_size'], self.args.crop_size)
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(filename))


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    file_os_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(file_os_dir)
    os.chdir('..')

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser.add_argument('--source_dataset', default='None', type=str, help='source dataset choice')
    arg_parser.add_argument('--flip', type=str2bool, default=False, help="flip")
    arg_parser.add_argument('--image_summary', type=str2bool, default=False, help="image_summary")
    arg_parser.add_argument('--save_prediction_to', type=str, default=None, help="where to save prediction results")

    args = arg_parser.parse_args()
    if args.split == "train": args.split = "val"
    if args.checkpoint_dir == "none": args.checkpoint_dir = args.pretrained_ckpt_file + "/eval"
    args, train_id, logger = init_args(args)
    args.batch_size_per_gpu = 2
    args.crop_size = args.target_crop_size
    args.base_size = args.target_base_size


    assert (args.source_dataset == 'synthia' and args.num_classes == 16) or (args.source_dataset == 'gta5' and args.num_classes == 19), 'dataset:{0:} - classes:{1:}'.format(args.source_dataset, args.num_classes)

    agent = Evaluater(args=args, cuda=True, train_id="train_id", logger=logger)
    agent.main()