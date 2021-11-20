import os
import torch

torch.backends.cudnn.benchmark = True
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from PIL import Image
from distutils.version import LooseVersion
import numpy as np
import sys

sys.path.append(os.path.abspath('.'))

from datasets.synthia_Dataset import SYNTHIA_Dataset
from datasets.gta5_Dataset import GTA5_Dataset
from datasets.cityscapes_Dataset import City_Dataset

from utils.losses import feat_reg_ST_loss
from tools.train_source import Trainer, str2bool, argparse, add_train_args, init_args, datasets_path

DEBUG = False


def memory_check(log_string):
    torch.cuda.synchronize()
    if DEBUG:
        print(log_string)
        print(' peak:', '{:.3f}'.format(torch.cuda.max_memory_allocated() / 1024 ** 3), 'GB')
        print(' current', '{:.3f}'.format(torch.cuda.memory_allocated() / 1024 ** 3), 'GB')


def add_UDA_train_args(arg_parser):
    # shared
    arg_parser.add_argument('--centroid_smoothing', default=-1, type=float, help="centroid smoothing coefficient, negative to disable")
    arg_parser.add_argument('--source_dataset', default='gta5', type=str, choices=['gta5', 'synthia'], help='source dataset choice')
    arg_parser.add_argument('--source_split', default='train', type=str, help='source datasets split')
    arg_parser.add_argument('--init_round', type=int, default=0, help='init_round')
    arg_parser.add_argument('--round_num', type=int, default=1, help="num round")
    arg_parser.add_argument('--epoch_each_round', type=int, default=2, help="epoch each round")
    arg_parser.add_argument('--logging_interval', type=int, default=1, help="interval in steps for logging")
    arg_parser.add_argument('--save_inter_model', type=str2bool, default=False, help="save model at the end of each epoch or not")

    # pseudo-labels
    arg_parser.add_argument('--no_uncertainty', type=str2bool, default=True, help="use uncertainty in the pesudo-label selection, default true'")

    # clustering
    arg_parser.add_argument('--lambda_things', default=1, type=float, help="things loss 权重")
    arg_parser.add_argument('--lambda_stuff', default=1, type=float, help="norm order of feature stuff loss")
    arg_parser.add_argument('--lambda_entropy', default=1, type=float, help="norm order of feature c2other loss")
    arg_parser.add_argument('--norm_order', default=1, type=int, help="norm order of feature clustering loss")
    arg_parser.add_argument('--thing_type', default='Entropy', type=str, choices=['Entropy', 'Squares','Cosine'], help='things alignment loss type choice')
    arg_parser.add_argument('--em_type', default='Entropy', type=str, choices=['Entropy', 'Squares'], help='em loss type choice')
    arg_parser.add_argument('--deque_capacity_factor', default=1., type=float, help="队列容量因子")

    return arg_parser


class UDATrainer(Trainer):

    def __init__(self, args, cuda=None, train_id="None", logger=None):
        super().__init__(args, cuda, train_id, logger)

        ### DATASETS ###
        self.logger.info('Adaptation {} -> {}'.format(self.args.source_dataset, self.args.target_dataset))

        source_data_kwargs = {'data_root_path': args.source_data_path,
                              'list_path': args.source_list_path,
                              'base_size': args.base_size,
                              'crop_size': args.crop_size}
        target_data_kwargs = {'data_root_path': args.data_root_path,
                              'list_path': args.list_path,
                              'base_size': args.target_base_size,
                              'crop_size': args.target_crop_size}
        dataloader_kwargs = {'batch_size': self.args.batch_size,
                             'num_workers': self.args.data_loader_workers,
                             'pin_memory': self.args.pin_memory,
                             'drop_last': True}

        if self.args.source_dataset == 'synthia':
            source_data_kwargs['class_16'] = target_data_kwargs['class_16'] = args.class_16

        source_data_gen = SYNTHIA_Dataset if self.args.source_dataset == 'synthia' else GTA5_Dataset

        if DEBUG: print('DEBUG: Loading training dataset (source)')
        source_dataset = source_data_gen(args, split='train', **source_data_kwargs)
        self.source_dataloader = data.DataLoader(source_dataset, shuffle=True, **dataloader_kwargs)
        self.source_train_iterations = (len(source_dataset) + self.args.batch_size) // self.args.batch_size
        if DEBUG: print('DEBUG: Loading validation dataset (source)')
        self.source_val_dataloader = data.DataLoader(source_data_gen(args, split='val', **source_data_kwargs), shuffle=False, **dataloader_kwargs)

        if DEBUG: print('DEBUG: Loading training dataset (target)')
        target_dataset = City_Dataset(args, split='train', **target_data_kwargs)
        self.target_dataloader = data.DataLoader(target_dataset, shuffle=True, **dataloader_kwargs)
        self.target_train_iterations = (len(target_dataset) + self.args.batch_size) // self.args.batch_size

        if DEBUG: print('DEBUG: Loading validation dataset (target)')
        target_data_set = City_Dataset(args, split='val', **target_data_kwargs)
        self.target_val_dataloader = data.DataLoader(target_data_set, shuffle=False, **dataloader_kwargs)

        # 由于val是调用的父类的validate(父类方法)，父类val使用的是self.dataloader.val_loader，所以这里需要用target_val_dataloader去覆盖。
        self.dataloader.val_loader = self.target_val_dataloader
        self.dataloader.valid_iterations = (len(target_data_set) + self.args.batch_size) // self.args.batch_size

        self.ignore_index = -1
        self.current_round = self.args.init_round
        self.round_num = self.args.round_num
        self.no_uncertainty = args.no_uncertainty

        ### LOSSES ###

        # ignore_index = -1, num_class = 19, deque_capacity_factor = 2.0, feat_channel = 2048, device = 'cuda'):
        self.feat_reg_ST_loss = feat_reg_ST_loss(ignore_index=-1,
                                                 num_class=self.args.num_classes,
                                                 deque_capacity_factor=self.args.deque_capacity_factor,
                                                 feat_channel=2048,
                                                 device=self.device)
        self.feat_reg_ST_loss.to(self.device)

        self.loss_kwargs = {}
        self.alignment_params = {'norm_order': args.norm_order, 'thing_type': args.thing_type, 'em_type': args.em_type}
        self.loss_kwargs['alignment_params'] = self.alignment_params

        self.best_MIou, self.best_iter, self.current_iter, self.current_epoch = None, None, None, None

        self.epoch_num = None

    def gen_pseudo_label(self):

        tqdm_epoch = tqdm(self.target_dataloader, total=self.target_train_iterations, desc="Generate pseudo label Epoch-{}-total-{}".format(self.current_epoch + 1, self.epoch_num))
        self.logger.info("Generate pseudo label...")
        self.model.eval()

        self.target_dataloader.dataset.random_mirror = False
        self.target_dataloader.dataset.gaussian_blur = False
        self.target_dataloader.dataset.color_jitter = False

        if not self.args.no_uncertainty:
            f_pass = 10
        else:
            f_pass = 1

        for image, _, id_gt in tqdm_epoch:
            if self.cuda:
                image = Variable(image).to(self.device)
            with torch.no_grad():
                cur_out_prob = []  # 输出的置信度
                for _ in range(f_pass):  # f_pass前向传播
                    outputs = self.model(image)[0]  # b,c,h,w
                    cur_out_prob.append(F.softmax(outputs, dim=1))  # 用于选择正伪标签
            if f_pass == 1:
                out_prob = cur_out_prob[0]
                max_value, max_idx = torch.max(out_prob, dim=1)  # 最大的预测值和对应的索引      b,h,w
                max_std = torch.zeros_like(max_value)
            else:
                out_prob = torch.stack(cur_out_prob)  # 当前批次的预测样本堆叠起来，N,b,c,h,w
                out_prob_std = torch.std(out_prob, dim=0)  # 正伪标签的方差      b,c,h,w
                out_prob_mean = torch.mean(out_prob, dim=0)  # 正伪标签的平均置信度
                max_value, max_idx = torch.max(out_prob_mean, dim=1)  # 最大的预测值和对应的索引      b,h,w
                max_std = out_prob_std.gather(1, max_idx.unsqueeze(dim=1)).squeeze(1)  # 最大预测值的方差         b,h,w

            thre_conf = []  # 置信度阈值, 长度为num_classes的list
            thre_std = []  # 方差阈值
            for i in range(self.args.num_classes):
                mask_i = (max_idx == i)
                mid = max_value[mask_i].size(0) // 2
                if mid == 0:
                    thre_conf.append(0)
                    thre_std.append(0)
                else:
                    # 计算置信度阈值
                    max_value_i, _ = torch.sort(max_value[mask_i])
                    thre_conf.append(min(0.9, max_value_i[mid]))

                    # 计算方差阈值
                    max_std_i, _ = torch.sort(max_std[mask_i])
                    thre_std.append(max(0.01, max_std_i[mid]))

            # 计算每一个点的阈值
            thre_conf = torch.tensor(thre_conf).cuda()[max_idx].detach()  # (num_classes) => (b,h,w)
            thre_std = torch.tensor(thre_std).cuda()[max_idx].detach()

            if not args.no_uncertainty:  # 选择满足置信度和不确定性条件的伪标签
                selected_idx = (max_value >= thre_conf) * (max_std < thre_std)  # b,h,w
            else:
                selected_idx = max_value >= thre_conf
            unselected_idx = ~selected_idx  # 取反，选择所以不满足条件的像素点

            pseudo_label = max_idx.clone().cpu()
            pseudo_label[unselected_idx] = self.ignore_index  # 未被选择的像素赋值为ignore_index

            batch_size = max_idx.size(0)
            for b in range(batch_size):
                pseudo_name = id_gt[b]
                label = pseudo_label[b]
                output = np.asarray(label, dtype=np.uint8)
                output = Image.fromarray(output)
                # output = colorize_mask(output, self.args.num_classes)
                save_path = self.pseudo_label_dir + pseudo_name
                save_dir = os.path.dirname(save_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                output.save(save_path)

    def updata_target_dataloader(self):

        target_data_kwargs = {'data_root_path': args.data_root_path,
                              'list_path': args.list_path,
                              'base_size': args.target_base_size,
                              'crop_size': args.target_crop_size,
                              'pseudo_floder': self.pseudo_label_dir,
                              'load_pseudo': True,
                              'pseudo_round': self.round_num}
        dataloader_kwargs = {'batch_size': self.args.batch_size,
                             'num_workers': self.args.data_loader_workers,
                             'pin_memory': self.args.pin_memory,
                             'drop_last': True,
                             }
        if DEBUG: print('DEBUG: Loading training pseudo dataset (target)')

        target_dataset = City_Dataset(self.args, split='train', **target_data_kwargs)
        self.target_dataloader = data.DataLoader(target_dataset, shuffle=True, **dataloader_kwargs)

    def main(self):  # train_round->train(父类方法)->train_one_epoch + validate(父类方法)
        # display command
        argv = sys.argv
        self.logger.info("command:")
        self.logger.info("python " + ' '.join(argv))
        # display args details
        self.logger.info("Global configuration as follows:")
        params = sorted(vars(self.args).items())
        for key, val in params:
            self.logger.info("{:25} {}".format(key, val))

        # choose cuda
        current_device = torch.cuda.current_device()
        self.logger.info("This model will run on {}".format(torch.cuda.get_device_name(current_device)))

        # load pretrained checkpoint
        if self.args.pretrained_ckpt_file is not None:
            if os.path.isdir(self.args.pretrained_ckpt_file):
                self.args.pretrained_ckpt_file = os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth')
            self.load_checkpoint(self.args.pretrained_ckpt_file)

        if not self.args.continue_training:
            self.best_MIou, self.best_iter, self.current_iter, self.current_epoch = 0, 0, 0, 0

        if self.args.continue_training:
            self.load_checkpoint(os.path.join(self.args.checkpoint_dir, self.train_id + 'final.pth'))

        self.args.iter_max = self.dataloader.num_iterations * self.args.epoch_each_round * self.round_num
        self.logger.info('Iter max: {} \nNumber of iterations: {}'.format(self.args.iter_max, self.dataloader.num_iterations))

        # train
        self.train_round()
        self.writer.close()

    def train_round(self):
        for r in range(self.current_round, self.round_num):
            self.logger.info("\n############## Begin {}/{} Round! #################\n".format(self.current_round + 1, self.round_num))
            self.logger.info("epoch_each_round: {}".format(self.args.epoch_each_round))

            self.epoch_num = (self.current_round + 1) * self.args.epoch_each_round
            self.pseudo_label_dir = os.path.join(args.data_root_path, self.exp_name, 'pseudo-label', str(self.current_round))
            if not os.path.exists(self.pseudo_label_dir):
                self.gen_pseudo_label()  # 生成伪标签
            else:
                self.logger.info(self.pseudo_label_dir + 'exists! Skip gen pseudo label')
            self.updata_target_dataloader()  # 更新目标域的dataloader

            self.train()

            self.current_round += 1

    def train_one_epoch(self):
        tqdm_epoch = tqdm(zip(self.source_dataloader, self.target_dataloader), total=self.dataloader.num_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}".format(self.current_round, self.current_epoch + 1, self.epoch_num))

        self.logger.info("Training one epoch...")
        self.Eval.reset()
        self.model.train()

        ### 日志设定 ###
        log_dic = {}

        batch_idx = 0
        for batch_s, batch_t in tqdm_epoch:
            self.poly_lr_scheduler(optimizer=self.optimizer, init_lr=self.args.lr)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)
            if self.current_iter < 1:
                memory_check('Start (step)')
            #######################
            # Source forward step #
            #######################

            # train data (labeled)
            x, y, _ = batch_s  # size:(B 3 H W) (B H W)
            if self.cuda:
                x, y = Variable(x).to(self.device), Variable(y).to(device=self.device, dtype=torch.long)

            if self.current_iter < 1:
                memory_check('Dataloader Source')

            ########################
            # model output ->  list of:  1) pred; 2) feat from encoder's output
            pred_and_feat = self.model(x)
            pred_source, feat_source = pred_and_feat  # size: (B  C  H  W)  (B  F  h  w)
            pred_source_softmax = F.softmax(pred_source, dim=1)  # size: (B  C  H  W)
            ########################

            if self.current_iter < 1:
                memory_check('Model Source')

            ##################################
            # Source supervised optimization #
            ##################################

            y = torch.squeeze(y, 1)  # size:(B H W) => (B 1 H W)
            loss = self.loss(pred_source, y)  # cross-entropy loss from train_source.py
            loss_ = loss  # 源域的交叉熵损失

            loss_.backward()

            # log
            log_dic['Source_ce_loss'] = loss.item()

            if self.current_iter < 1:
                memory_check('End Source')

            #######################
            # Target forward step #
            #######################

            # target data (unlabeld)
            x, p_y, _ = batch_t  # 图片、伪标签、文件名
            if self.cuda:
                x, p_y = Variable(x).to(self.device), Variable(p_y).to(device=self.device, dtype=torch.long)

            if self.current_iter < 1:
                memory_check('Dataloader Target')

            ########################
            pred_and_feat = self.model(x)  # model output ->  list of:  1) pred; 2) feat from encoder's output
            pred_target, feat_target = pred_and_feat  # size: (B  C  H  W)  (B  F  h  w)
            pred_target_softmax = F.softmax(pred_target, dim=1)  # size: (B  C  H  W)
            ########################

            if self.current_iter < 1:
                memory_check('Model Target')

            # 临时可视化标签图片
            # max_value, max_idx = torch.max(F.softmax(pred_target, dim=1), dim=1)
            # output = np.asarray(max_idx.clone().cpu(), dtype=np.uint8)
            # output = colorize_mask(output[0], self.args.num_classes)
            # output.save("/home/haol/pred_target_{}.png".format(batch_idx))

            #####################
            # Adaptation Losses #
            #####################
            # Set some inputs to the adaptation modules
            self.loss_kwargs['source_prob'] = pred_source_softmax  # 源域预测结果softmax,
            self.loss_kwargs['target_prob'] = pred_target_softmax  # 目标域预测结果softmax
            self.loss_kwargs['source_feat'] = feat_source  # 源域的中间特征
            self.loss_kwargs['target_feat'] = feat_target  # 目标域的中间特征
            self.loss_kwargs['source_label'] = y  # 源域的标签
            self.loss_kwargs['target_label'] = p_y  # 目标域的伪标签
            self.loss_kwargs['smo_coeff'] = args.centroid_smoothing  # 平均指数移动的参数

            # 传入参数，计算当前批次的loss
            loss_dict = self.feat_reg_ST_loss(**self.loss_kwargs)
            stuff_alignment_loss, thing_alignment_loss, EM_loss = loss_dict['stuff_alignment_loss'], loss_dict['thing_alignment_loss'], loss_dict['EM_loss']

            thing_alignment_loss = self.args.lambda_things * thing_alignment_loss
            stuff_alignment_loss = self.args.lambda_stuff * stuff_alignment_loss
            em_loss = self.args.lambda_entropy * EM_loss

            total_loss = thing_alignment_loss + stuff_alignment_loss + em_loss
            total_loss.backward()

            # 保存当前loss

            log_dic['Stuff_alignment_loss'] = stuff_alignment_loss.item()
            log_dic['Thing_alignment_loss'] = thing_alignment_loss.item()
            log_dic['EM_loss'] = em_loss.item()
            log_dic['Target_loss'] = total_loss.item()
            log_string = 'epoch{}-batch-{}:' + '={:3f}-'.join(log_dic.keys()) + '={:3f}'

            # logging
            if batch_idx % self.args.logging_interval == 0:
                self.logger.info(log_string.format(self.current_epoch, batch_idx, *log_dic.values()))
                for name, elem in log_dic.items():
                    self.writer.add_scalar(name, elem, self.current_iter)

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.current_iter < 1:
                memory_check('End (step)')

            batch_idx += 1
            self.current_iter += 1

        tqdm_epoch.close()
        self.feat_reg_ST_loss.reset()
        # eval on source domain
        # self.validate_source()

        if self.args.save_inter_model:
            self.logger.info("Saving model of epoch {} ...".format(self.current_epoch))
            self.save_checkpoint(self.train_id + '_epoch{}.pth'.format(self.current_epoch))


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    file_os_dir = os.path.dirname(os.path.abspath(__file__))  # ./tools路径
    os.chdir(file_os_dir)  # 工作目录切换到./tools路径
    os.chdir('..')  # 工作目录切换到./tools路径的上一级目录

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser = add_UDA_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)
    args.source_data_path = datasets_path[args.source_dataset]['data_root_path']
    args.source_list_path = datasets_path[args.source_dataset]['list_path']

    args.target_dataset = args.dataset

    train_id = str(args.source_dataset) + "2" + str(args.target_dataset)

    assert (args.source_dataset == 'synthia' and args.num_classes == 16) or (args.source_dataset == 'gta5' and args.num_classes == 19), 'dataset:{0:} - classes:{1:}'.format(args.source_dataset, args.num_classes)

    agent = UDATrainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()
