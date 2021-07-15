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
import collections
import sys

sys.path.append(os.path.abspath('.'))

from datasets.synthia_Dataset import SYNTHIA_Dataset
from datasets.gta5_Dataset import GTA5_Dataset
from datasets.cityscapes_Dataset import City_Dataset
from datasets.cityscapes_Dataset import colorize_mask

# from utils.losses import feat_reg_ST_loss, IW_MaxSquareloss
from utils.losses_modified import feat_reg_ST_loss, IW_MaxSquareloss
from tools.train_source import Trainer, str2bool, argparse, add_train_args, init_args, datasets_path

DEBUG = False


def memory_check(log_string):
    torch.cuda.synchronize()
    if DEBUG:
        print(log_string)
        print(' peak:', '{:.3f}'.format(torch.cuda.max_memory_allocated() / 1024 ** 3), 'GB')
        print(' current', '{:.3f}'.format(torch.cuda.memory_allocated() / 1024 ** 3), 'GB')


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

        self.dataloader.val_loader = self.target_val_dataloader
        self.dataloader.valid_iterations = (len(target_data_set) + self.args.batch_size) // self.args.batch_size

        self.ignore_index = -1
        self.current_round = self.args.init_round
        self.round_num = self.args.round_num
        self.no_uncertainty = args.no_uncertainty

        ### LOSSES ###
        self.feat_reg_ST_loss = feat_reg_ST_loss(ignore_index=-1,
                                                 num_class=self.args.num_classes,
                                                 device=self.device)
        self.feat_reg_ST_loss.to(self.device)

        self.use_em_loss = self.args.lambda_entropy != 0.
        if self.use_em_loss:
            self.entropy_loss = IW_MaxSquareloss(ignore_index=-1,
                                                 num_class=self.args.num_classes,
                                                 ratio=self.args.IW_ratio)
            self.entropy_loss.to(self.device)

        self.loss_kwargs = {}
        self.alignment_params = {'norm_order': args.norm_order}
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

        if not self.args.no_uncertainty and self.current_round > 0:
            f_pass = 10
        else:
            f_pass = 1

        for image, _, id_gt in tqdm_epoch:
            if self.cuda:
                image = Variable(image).to(self.device)
            with torch.no_grad():
                cur_out_prob = []  # 输出的置信度
                cur_out_prob_nl = []  # 带温度系数的出的置信度
                for _ in range(f_pass):  # f_pass前向传播
                    outputs = self.model(image)[0]  # b,c,h,w
                    cur_out_prob.append(F.softmax(outputs, dim=1))  # 用于选择正伪标签
                    # cur_out_prob_nl.append(F.softmax(outputs / args.temp_nl, dim=1))  # 用于选择负伪标签
            if f_pass == 1:
                out_prob = cur_out_prob[0]
                max_value, max_idx = torch.max(out_prob, dim=1)  # 最大的预测值和对应的索引      b,h,w
                max_std = torch.zeros_like(max_value)
            else:
                out_prob = torch.stack(cur_out_prob)  # 当前批次的预测样本堆叠起来，N,b,c,h,w
                out_std = torch.std(out_prob, dim=0)  # 正伪标签的方差      b,c,h,w
                out_prob_mean = torch.mean(out_prob, dim=0)  # 正伪标签的平均置信度
                max_value, max_idx = torch.max(out_prob_mean, dim=1)  # 最大的预测值和对应的索引      b,h,w
                max_std = out_std.gather(1, max_idx.unsqueeze(dim=1)).squeeze(1)  # 最大预测值的方差         b,h,w

            # out_prob_nl = torch.stack(cur_out_prob_nl)
            # out_std_nl = torch.std(out_prob_nl, dim=0)  # 负伪标签的方差      b,c,h,w
            # out_prob_nl_mean = torch.mean(out_prob_nl, dim=0)  # 负伪标签的平均置信度
            # min_value, min_idx = torch.min(out_prob_nl_mean, dim=1)  # 最小的预测值和对应的索引      b,h,w
            # min_std = out_std_nl.gather(1, min_idx.unsqueeze(dim=1)).squeeze(1)  # 最小预测值的方差         b,h,w

            def gen_label(self, value, idx, std, conf_thre, uncert_thre, is_nl, ignore_index):

                if is_nl:
                    save = self.n_pseudo_label_dir
                    if not args.no_uncertainty:  # 选择满足置信度和不确定性条件的负伪标签
                        selected_idx = (value <= conf_thre) * (std < uncert_thre)  # b,h,w
                    else:
                        selected_idx = value <= args.conf_thre
                else:
                    save = self.pseudo_label_dir

                    if not args.no_uncertainty:  # 选择满足置信度和不确定性条件的伪标签
                        selected_idx = (value >= conf_thre) * (std < uncert_thre)  # b,h,w
                    else:
                        selected_idx = value >= args.conf_thre
                unselected_idx = ~selected_idx  # 取反，选择所以不满足条件的像素点

                pseudo_label = idx.clone().cpu()
                pseudo_maxstd = std.clone().cpu()

                pseudo_label[unselected_idx] = 255  # 未被选择的像素赋值为255
                pseudo_maxstd[unselected_idx] = 1000  # 未被选择的像素的方差赋值一个较大的方差

                # max_class_len = [0.01] * self.args.num_classes  # 源域中每个类别平均占比先验信息
                # ######
                # if args.class_blnc and self.current_epoch < args.class_blnc - 1:  # 如果选择类别平衡，并且当前迭代次数少于最大需要类别平衡的迭代次数
                #     for class_idx in range(self.args.num_classes):
                #         cur_max_class_len = int(max_class_len[class_idx] * 512 * 1024)  # 当前类别平均像素点的先验信息
                #         class_pos = np.where(pseudo_label == class_idx)  # 当前类别在伪标签中的位置
                #         class_maxstd = pseudo_maxstd[class_pos]  # 当前类别的方差
                #         class_maxstd_sort = np.sort(class_maxstd)  # 排序后的方差
                #
                #         class_len = len(class_pos[0])  # 当前类别像素点的数量
                #         if class_len > 0:
                #             class_len_blnc = min(cur_max_class_len, class_len)  # 类别平衡后的像素点的数量
                #
                #             class_maxstd_sort_max = class_maxstd_sort[class_len_blnc - 1]  # 类别平衡筛选后的最大方差
                #             unselected_idx = pseudo_maxstd > class_maxstd_sort_max  # 选取大于方差的idx
                #             pseudo_label[unselected_idx] = self.ignore_index  # 大于方差的idx分配为忽略的标签

                for b in range(self.args.batch_size):
                    pseudo_name = id_gt[b]
                    label = pseudo_label[b]
                    output = np.asarray(label, dtype=np.uint8)
                    output = Image.fromarray(output)
                    # output = colorize_mask(output, self.args.num_classes)
                    save_path = save + pseudo_name
                    save_dir = os.path.dirname(save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    output.save(save_path)

            # 生成正伪标签
            gen_label(self, max_value, max_idx, max_std, args.tau_p, args.kappa_p, is_nl=False, ignore_index=self.ignore_index)
            # 生成负伪标签
            # gen_label(min_value, min_idx, min_std, args.tau_n, args.kappa_n, is_nl=True)

    def updata_target_dataloader(self):

        target_data_kwargs = {'data_root_path': args.data_root_path,
                              'list_path': args.list_path,
                              'base_size': args.target_base_size,
                              'crop_size': args.target_crop_size,
                              'pseudo_floder': self.pseudo_label_dir,
                              'load_pseudo': True}
        dataloader_kwargs = {'batch_size': self.args.batch_size,
                             'num_workers': self.args.data_loader_workers,
                             'pin_memory': self.args.pin_memory,
                             'drop_last': True,
                             'pseudo_round':self.round_num}
        if DEBUG: print('DEBUG: Loading training pseudo dataset (target)')

        target_dataset = City_Dataset(self.args, split='train', **target_data_kwargs)
        self.target_dataloader = data.DataLoader(target_dataset, shuffle=True, **dataloader_kwargs)

    def main(self):
        # display args details
        self.logger.info("Global configuration as follows:")
        for key, val in vars(self.args).items():
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
            self.n_pseudo_label_dir = os.path.join(args.data_root_path, self.exp_name, 'negative-pseudo-label', str(self.current_round))
            self.pseudo_label_dir = os.path.join(args.data_root_path, self.exp_name, 'pseudo-label', str(self.current_round))
            if self.current_round > 0:
                self.gen_pseudo_label()
            self.updata_target_dataloader()

            self.train()

            self.current_round += 1

    def train_one_epoch(self):
        tqdm_epoch = tqdm(zip(self.source_dataloader, self.target_dataloader), total=self.dataloader.num_iterations,
                          desc="Train Round-{}-Epoch-{}-total-{}".format(self.current_round, self.current_epoch + 1, self.epoch_num))

        self.logger.info("Training one epoch...")
        self.Eval.reset()

        # Set the model to be in training mode (for batchnorm and dropout)
        if self.args.freeze_bn:  # default False
            self.model.eval()
            self.logger.info("freeze bacth normalization successfully!")
        else:
            self.model.train()

        ### Logging setup ###
        log_list, log_strings = [None], ['Source_loss_ce']

        log_list += [None] * 3
        log_strings += ['thing_alignment_loss', 'stuff_alignment_loss', 'contrastive_loss']

        if self.use_em_loss:
            log_strings.append('EM_loss')
            log_list.append(None)

        log_string = 'epoch{}-batch-{}:' + '={:3f}-'.join(log_strings) + '={:3f}'

        batch_idx = 0
        for batch_s, batch_t in tqdm_epoch:
            self.poly_lr_scheduler(optimizer=self.optimizer, init_lr=self.args.lr)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]["lr"], self.current_iter)

            if self.current_iter < 1: memory_check('Start (step)')

            #######################
            # Source forward step #
            #######################

            # train data (labeled)
            x, y, _ = batch_s
            if self.cuda:
                x, y = Variable(x).to(self.device), Variable(y).to(device=self.device, dtype=torch.long)

            if self.current_iter < 1: memory_check('Dataloader Source')

            ########################
            # model output ->  list of:  1) pred; 2) feat from encoder's output
            pred_and_feat = self.model(x)
            pred_source, feat_source = pred_and_feat
            ########################

            if self.current_iter < 1: memory_check('Model Source')

            # max_value, max_idx = torch.max(F.softmax(pred_source, dim=1), dim=1)
            # output = np.asarray(max_idx.clone().cpu(), dtype=np.uint8)
            # output = colorize_mask(output[0], self.args.num_classes)
            # output.save("/home/haol/y_s_{}.png".format(batch_idx))
            ##################################
            # Source supervised optimization #
            ##################################

            y = torch.squeeze(y, 1)
            loss = self.loss(pred_source, y)  # cross-entropy loss from train_source.py
            loss_ = loss

            loss_.backward(retain_graph=True)

            # log
            log_ind = 0
            log_list[log_ind] = loss.item()
            log_ind += 1

            if self.current_iter < 1: memory_check('End Source')

            #######################
            # Target forward step #
            #######################

            # target data (unlabeld)
            x, p_y, _ = batch_t
            if self.cuda:
                # x = Variable(x).to(self.device)
                x, p_y = Variable(x).to(self.device), Variable(p_y).to(device=self.device, dtype=torch.long)

            if self.current_iter < 1: memory_check('Dataloader Target')
            #
            # output = np.asarray(p_y.clone().cpu(), dtype=np.uint8)
            # output = colorize_mask(output[0], self.args.num_classes)
            # output.save("/home/haol/p_y_{}.png".format(batch_idx))

            ########################
            # model output ->  list of:  1) pred; 2) feat from encoder's output
            pred_and_feat = self.model(x)  # creates the graph
            pred_target, feat_target = pred_and_feat
            ########################

            if self.current_iter < 1: memory_check('Model Target')

            # max_value, max_idx = torch.max(F.softmax(pred_target, dim=1), dim=1)
            # output = np.asarray(max_idx.clone().cpu(), dtype=np.uint8)
            # output = colorize_mask(output[0], self.args.num_classes)
            # output.save("/home/haol/pred_target_{}.png".format(batch_idx))
            #####################
            # Adaptation Losses #
            #####################

            # Set some inputs to the adaptation modules
            self.loss_kwargs['source_prob'] = F.softmax(pred_source, dim=1)
            self.loss_kwargs['target_prob'] = F.softmax(pred_target, dim=1)
            self.loss_kwargs['source_feat'] = feat_source
            self.loss_kwargs['target_feat'] = feat_target
            self.loss_kwargs['source_gt'] = y
            self.loss_kwargs['target_gt'] = p_y
            self.loss_kwargs['smo_coeff'] = args.centroid_smoothing

            # Pass the input dict to the adaptation full loss
            loss_dict = self.feat_reg_ST_loss(**self.loss_kwargs)

            c2c_dist_B, i2c_dist_F, i2i_dist_F, c2other_dist = loss_dict['c2c_dist_B'], loss_dict['i2c_dist_F'], loss_dict['i2i_dist_F'], loss_dict['c2other_dist']
            log_strings += ['thing_alignment_loss', 'stuff_alignment_loss', 'contrastive_loss']
            thing_alignment_loss = self.args.lambda_things * c2c_dist_B
            stuff_alignment_loss = self.args.lambda_stuff * (i2c_dist_F + i2i_dist_F)

            contrastive_loss = thing_alignment_loss + stuff_alignment_loss - self.args.lambda_c2other * c2other_dist
            retain_graph = self.use_em_loss
            if contrastive_loss.item() != 0:
                contrastive_loss.backward(retain_graph=retain_graph)
            if self.current_iter < 1: memory_check('contrastive_loss Loss')
            # log
            log_list[log_ind:log_ind + 3] = [thing_alignment_loss.item(), stuff_alignment_loss.item(), contrastive_loss.item()]
            log_ind += 3

            if self.use_em_loss:
                em_loss = self.args.lambda_entropy * self.entropy_loss(pred_target, F.softmax(pred_target, dim=1))
                em_loss.backward()
                if self.current_iter < 1: memory_check('Entropy Loss')
                # log
                log_list[log_ind] = em_loss.item()
                log_ind += 1

            self.optimizer.step()
            self.optimizer.zero_grad()

            # logging
            if batch_idx % self.args.logging_interval == 0:
                self.logger.info(log_string.format(self.current_epoch, batch_idx, *log_list))
                for name, elem in zip(log_strings, log_list):
                    self.writer.add_scalar(name, elem, self.current_iter)

            batch_idx += 1

            self.current_iter += 1

            if self.current_iter < 1: memory_check('End (step)')

        tqdm_epoch.close()

        # eval on source domain
        # self.validate_source()

        if self.args.save_inter_model:
            self.logger.info("Saving model of epoch {} ...".format(self.current_epoch))
            self.save_checkpoint(self.train_id + '_epoch{}.pth'.format(self.current_epoch))


def add_UDA_train_args(arg_parser):
    # shared
    arg_parser.add_argument('--use_source_gt', default=False, type=str2bool, help='use source label or segmented image for pixel/feature classification')
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
    arg_parser.add_argument('--tau-p', default=0.70, type=float, help='confidece threshold for positive pseudo-labels, default 0.70')
    arg_parser.add_argument('--tau-n', default=0.05, type=float, help='confidece threshold for negative pseudo-labels, default 0.05')
    arg_parser.add_argument('--kappa-p', default=0.05, type=float, help='uncertainty threshold for positive pseudo-labels, default 0.05')
    arg_parser.add_argument('--kappa-n', default=0.005, type=float, help='uncertainty threshold for negative pseudo-labels, default 0.005')
    arg_parser.add_argument('--temp-nl', default=2.0, type=float, help='temperature for generating negative pseduo-labels, default 2.0')

    # clustering
    # arg_parser.add_argument('--lambda_cluster', default=0., type=float, help="lambda of clustering loss")
    # arg_parser.add_argument('--lambdas_cluster', default=None, type=str, help="lambda intra-domain source, lambda intra-domain target, lambda inter-domain")
    arg_parser.add_argument('--lambda_things', default=1, type=float, help="norm order of feature things loss")
    arg_parser.add_argument('--lambda_stuff', default=1, type=float, help="norm order of feature stuff loss")
    arg_parser.add_argument('--lambda_c2other', default=1, type=float, help="norm order of feature c2other loss")
    arg_parser.add_argument('--norm_order', default=1, type=int, help="norm order of feature clustering loss")

    # orthogonality
    # arg_parser.add_argument('--lambda_ortho', default=0., type=float, help="lambda of orthogonality loss")
    # arg_parser.add_argument('--ortho_temp', default=1., type=float, help="temperature for similarity based-distribution")

    # sparsity
    # arg_parser.add_argument('--lambda_sparse', default=0., type=float, help="lambda of sparsity loss")
    # arg_parser.add_argument('--sparse_norm_order', default=2., type=float, help="sparsity loss exponent")
    # arg_parser.add_argument('--sparse_rho', default=0.5, type=float, help="sparsity loss constant threshold")

    # off-the-shelf entropy loss
    arg_parser.add_argument('--lambda_entropy', type=float, default=0., help="lambda of target loss")
    arg_parser.add_argument('--IW_ratio', type=float, default=0.2, help='the ratio of image-wise weighting factor')

    return arg_parser


def init_UDA_args(args):
    def str2none(l):
        l = [l] if not isinstance(l, list) else l
        for i, el in enumerate(l):
            if el == 'None':
                l[i] = None
        return l if len(l) > 1 else l[0]

    def str2float(l):
        for i, el in enumerate(l):
            try:
                l[i] = float(el)
            except (ValueError, TypeError):
                l[i] = el
        return l

    return args


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'

    file_os_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(file_os_dir)
    os.chdir('..')

    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)
    arg_parser = add_UDA_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id, logger = init_args(args)
    args = init_UDA_args(args)
    args.source_data_path = datasets_path[args.source_dataset]['data_root_path']
    args.source_list_path = datasets_path[args.source_dataset]['list_path']

    args.target_dataset = args.dataset

    train_id = str(args.source_dataset) + "2" + str(args.target_dataset)

    assert (args.source_dataset == 'synthia' and args.num_classes == 16) or (args.source_dataset == 'gta5' and args.num_classes == 19), 'dataset:{0:} - classes:{1:}'.format(args.source_dataset, args.num_classes)

    agent = UDATrainer(args=args, cuda=True, train_id=train_id, logger=logger)
    agent.main()
