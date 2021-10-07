import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import label as sklabel


class feat_reg_ST_loss(nn.Module):

    def __init__(self, ignore_index=-1, num_class=19, pool_capacity=50, feat_channel=2048, device='cuda'):
        super(feat_reg_ST_loss, self).__init__()

        if num_class == 19:  # GTA5
            self.BG_LABEL = [0, 1, 2, 3, 4, 8, 9, 10]
            self.FG_LABEL = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]

        else:  # SYNTHIA
            self.BG_LABEL = [0, 1, 2, 3, 4, 8, 10]
            self.FG_LABEL = [5, 6, 7, 11, 12, 13, 15, 17, 18]

        self.ignore_index = ignore_index
        self.first_run = True
        self.device = device
        self.num_class = num_class

        self.centroids_source_FG = None
        self.centroids_source_BG = None
        self.centroids_target_BG = None
        self.centroids_source_FG_avg = None
        self.centroids_source_BG_avg = None
        self.centroids_target_BG_avg = None

        self.source_feat = None
        self.source_gt = None
        self.target_feat = None
        self.target_gt = None

        self.dist_func = None  # L0、L1、L2...

        self.B = None  # size of the current batch
        self.F = feat_channel  # feat channel, size F
        self.C_bg, self.C_fg = len(self.BG_LABEL), len(self.FG_LABEL)  # number of BG abd FG

        self.Hs, self.Ws, self.hs, self.ws = None, None, None, None
        self.Ht, self.Wt, self.ht, self.wt = None, None, None, None

        self.pool_capacity = pool_capacity  # The capacity of the source instance feature pool
        self.pool_ptr = [0] * self.C_fg  # source instance feature pool pointer
        self.ins_feat_pool_source = torch.full(dtype=torch.float32,  # source instance feature pool
                                               fill_value=float("Inf"),  # size C_FB x p_capacity x F
                                               size=(self.C_fg, pool_capacity, self.F)).to(device)

    ## PRE-PROCESSING ###
    def dist(self, tensor, dim=None):
        if isinstance(self.dist_func, int):
            return torch.norm(tensor, p=self.dist_func, dim=dim) / tensor.numel()  # sum(abs(tensor)**p)**(1./p) , default output is scalar
        else:
            return self.dist_func(tensor)

    def feature_processing(self, feat, softmax, gt, domain, argmax_dws_type='bilinear'):
        """
        Process feature and softmax tensors in order to get downsampled and 2D-shaped representations
        :param feat: feature tensor (B x F x h x w)
        :param softmax: softmax map (B x C x H x W)
        :param gt: Ground truth (B x H x W)
        :param domain: source or target
        :param argmax_dws_type: direct 'nearest' or 'bilinear' through from softmax
        """
        self.B = softmax.size(0)
        assert self.F == feat.size(1)
        # 获取源域和目标域的特征图、预测结果的尺度
        if domain == 'source':
            self.Hs, self.Ws, self.hs, self.ws = softmax.size(2), softmax.size(3), feat.size(2), feat.size(3)
        else:
            self.Ht, self.Wt, self.ht, self.wt = softmax.size(2), softmax.size(3), feat.size(2), feat.size(3)
        h, w = feat.size(2), feat.size(3)

        feat = feat.permute(0, 2, 3, 1).contiguous()  # size B x h x w x F
        feat = feat.view(-1, feat.size()[-1])  # size N x F (N = B x h x w)

        # _, argmax = torch.max(softmax, dim=1)  # size B x H x W

        # if argmax_dws_type == 'nearest':
        #     right_pre_dws = torch.squeeze(F.interpolate(torch.unsqueeze(argmax.float(), dim=1), size=(h, w), mode='nearest'), dim=1)  # size B x h x w
        # if argmax_dws_type == 'bilinear':
        #     softmax_dws = F.interpolate(softmax, size=(h, w), mode='bilinear', align_corners=True)  # size B x C x h x w
        #     _, right_pre_dws = torch.max(softmax_dws, dim=1)  # size B x h x w

        gt_dws = torch.squeeze(F.interpolate(torch.unsqueeze(gt.float(), dim=1), size=(h, w), mode='nearest'), dim=1)  # size B x h x w， GT下采样到和特征图一样大

        # Only compute the correct prediction map
        # right_pre_dws[right_pre_dws != gt_dws] = self.ignore_index  # size B x h x w
        # right_pre_dws = right_pre_dws.view(-1)  # size N
        gt_dws = gt_dws.view(-1)    # size N（N = B x h x w）

        if domain == 'source':
            self.source_feat = feat  # N x F, feat vector of each pixel
            self.source_gt = gt_dws  # N , right prediction of each pixel
        else:
            self.target_feat = feat  # N x F
            self.target_gt = gt_dws  # N

    ## AUXILIARY LOSS METHODS ##
    def get_instances(self, label_mask):
        """Compute the all instances Ids and area of batch samples in the given mask
        Parameters
        ----------
        label_mask: tensor (B x h x w)
            mask of a certain category

        Returns
        -------
        ins_Ids: tensor N (N = B x h x w)
            instances Ids, (start Id: 1)
        ins_area: list
           area((number of pixels)) of each instance

        Examples
        --------
        input:
            label_mask = tensor([[[False, True,  False],
                                 [False,  False, False],
                                 [False,  True, True]],

                                [[ True, False,  True],
                                 [False, False,  True],
                                 [False, False,  True]]])
        output:
            ins_Ids = tensor([[[0, 1, 0],
                                    [0, 0, 0],
                                    [0, 2, 2]],

                                   [[3, 0, 4],
                                    [0, 0, 4],
                                    [0, 0, 4]]])
            ins_area = [1, 2, 1, 3]

        """
        # number of all instances
        ins_num_total = 0
        ins_Ids = None
        # compute instances for each batch of samples
        for i in range(self.B):
            label_mask_i = label_mask[i]
            if torch.sum(label_mask_i) > 0:
                label_mask_np = label_mask_i.cpu().numpy().astype(int)
                # compute instance Id and number of all instances (a connected component is an instance)
                ins_Ids_np, ins_num = sklabel(label_mask_np, background=0, return_num=True, connectivity=2)
                # select only Id of instances, Id 0 is non-instance
                ins_Ids_np_mask_i = ins_Ids_np != 0
                # add the current instances number to instance Ids
                ins_Ids_np[ins_Ids_np_mask_i] += ins_num_total

                ins_mask_i = torch.LongTensor(ins_Ids_np).to(self.device)
                ins_num_total += ins_num
            else:
                ins_mask_i = label_mask_i.long().to(self.device)

            if i == 0:
                ins_Ids = ins_mask_i.unsqueeze(0)
            else:
                ins_Ids = torch.cat([ins_Ids, ins_mask_i.unsqueeze(0)])

        # compute instance area
        ins_area = np.zeros(ins_num_total, dtype=int)
        for i in range(ins_num_total):
            ins_area[i] = torch.sum(ins_Ids == (i + 1)).item()

        ins_Ids = ins_Ids.view(-1)
        return ins_Ids, ins_area

    def computer_stuff(self, centroids_smoothing=-1):
        """
        for BG(stuff), compute the BG centroid of source and target
        :param centroids_smoothing: if > 0 new centroids are updated over avg past ones
        """
        centroid_list_source_BG, centroid_list_target_BG = [], []

        ## 计算源域和目标域的背景（stuff）的中心
        for label_i in self.BG_LABEL:
            # boolean tensor, True where features belong to class label_i
            source_mask = torch.eq(self.source_gt.detach(), label_i)  # size N
            target_mask = torch.eq(self.target_gt.detach(), label_i)  # size N

            # select only features of class label_i
            source_feat_i = self.source_feat[source_mask, :]  # size Ns_i x F
            target_feat_i = self.target_feat[target_mask, :]  # size Nt_i x F

            # compute the centroid of source
            if source_feat_i.size(0) > 0:
                centroid = torch.mean(source_feat_i, dim=0, keepdim=True)  # size 1 x F
                centroid_list_source_BG.append(centroid)
            else:
                centroid = torch.tensor([[float("Inf")] * self.F], dtype=torch.float).to(self.device)  # size 1 x F
                centroid_list_source_BG.append(centroid)

            # compute the centroid of target
            if target_feat_i.size(0) > 0:
                centroid = torch.mean(target_feat_i, dim=0, keepdim=True)  # size 1 x F
                centroid_list_target_BG.append(centroid)
            else:
                centroid = torch.tensor([[float("Inf")] * self.F], dtype=torch.float).to(self.device)  # size 1 x F
                centroid_list_target_BG.append(centroid)

        self.centroids_source_BG = torch.squeeze(torch.stack(centroid_list_source_BG, dim=0))  # size C_BG x 1 x F -> C_BG x F
        self.centroids_target_BG = torch.squeeze(torch.stack(centroid_list_target_BG, dim=0))  # size C_BG x 1 x F -> C_BG x F

        # 指数平均移动
        if centroids_smoothing >= 0.:
            if self.centroids_source_BG_avg is None: self.centroids_source_BG_avg = self.centroids_source_BG  # size C_BG x F
            # In early steps there may be no centroids for small classes, so avoid averaging with Inf values by replacing them with values of current step
            self.centroids_source_BG_avg = torch.where(self.centroids_source_BG_avg != float('inf'), self.centroids_source_BG_avg, self.centroids_source_BG)
            # In some steps there may be no centroids for some classes, so avoid averaging with Inf values by replacing them with avg values
            self.centroids_source_BG = torch.where(self.centroids_source_BG == float('inf'), self.centroids_source_BG_avg.detach(), self.centroids_source_BG)
            # Exponential Moving Average
            self.centroids_source_BG = centroids_smoothing * self.centroids_source_BG + (1 - centroids_smoothing) * self.centroids_source_BG_avg.detach()
            self.centroids_source_BG_avg = self.centroids_source_BG.detach().clone()

            if self.centroids_target_BG_avg is None: self.centroids_target_BG_avg = self.centroids_target_BG  # size C_BG x F
            self.centroids_target_BG_avg = torch.where(self.centroids_target_BG_avg != float('inf'), self.centroids_target_BG_avg, self.centroids_target_BG)
            self.centroids_target_BG = torch.where(self.centroids_target_BG == float('inf'), self.centroids_target_BG_avg.detach(), self.centroids_target_BG)
            self.centroids_target_BG = centroids_smoothing * self.centroids_target_BG + (1 - centroids_smoothing) * self.centroids_target_BG_avg.detach()
            self.centroids_target_BG_avg = self.centroids_target_BG.detach().clone()

    def computer_things(self, centroids_smoothing=-1):
        """
        对于前景（things），计算每一个前景的中心，
         for FG(things), compute the FG centroid of source, instance feat list of target and source instance feature pool
        :param centroids_smoothing: if > 0 new centroids are updated over avg past ones

        """
        centroid_list_source_FG = []
        ins_feat_list_target = [[] for _ in range(len(self.FG_LABEL))]  # list size: C_FB x N_ins_i x F，每执行一次都会清空

        for i, label_i in enumerate(self.FG_LABEL):

            ## 计算源域的类别label_i的特征向量中心
            source_mask = torch.eq(self.source_gt.detach(), label_i)  # size N，类别label_i的掩码
            target_mask = torch.eq(self.target_gt.detach(), label_i)  # size N

            source_feat_i = self.source_feat[source_mask, :]  # size Ns_i x F， 类别label_i的特征向量
            # compute the FG centroid  of source
            if source_feat_i.size(0) > 0:
                centroid = torch.mean(source_feat_i, dim=0, keepdim=True)  # size 1 x F 类别label_i的平均特征向量
                centroid_list_source_FG.append(centroid)
            else:
                centroid = torch.tensor([[float("Inf")] * self.F], dtype=torch.float).to(self.device)  # size 1 x F
                centroid_list_source_FG.append(centroid)
            #################

            ## 计算源域和目标域的类别label_i的实例特征池
            source_mask_full = source_mask.view(self.B, self.hs, self.ws)  # size N -> B x h x w
            target_mask_full = target_mask.view(self.B, self.ht, self.wt)  # size N -> B x h x w

            # Compute the instances Ids and area of class i，利用连通区域算法计算每一个实例的面积，以及对应的编号
            source_ins_Ids, source_ins_area = self.get_instances(source_mask_full)  # size N
            target_ins_Ids, target_ins_area = self.get_instances(target_mask_full)  # size N


            # descending order of source instance area
            sort_argmax = np.argsort(source_ins_area)[::-1]
            # compute source instance feature pool
            for j in range(min(10, len(sort_argmax))):  # 遍历面积最大的前10个实例
                # current instance Id (instance Id stard with 0!!!!)
                ins_Id_j = sort_argmax[j] + 1

                # boolean tensor, True where features belong to instance ins_Id_j
                ins_mask_j = source_ins_Ids == ins_Id_j  # size N

                # select only features of instance ins_Id_j
                ins_feat_j = self.source_feat[ins_mask_j, :]
                ins_feat_j_avg = torch.mean(ins_feat_j, dim=0, keepdim=True)  # size 1 x F

                # compute the index of the instance feature in the source instance feature pool
                pos = int(self.pool_ptr[i] % self.pool_capacity)        # 计算在特征池中的位置
                # put in the pool
                self.ins_feat_pool_source[i, pos, :] = ins_feat_j_avg.detach().clone()  # 加入特征池
                self.pool_ptr[i] += 1

            # descending order of target instance area
            sort_argmax = np.argsort(target_ins_area)[::-1]
            # compute target instance feature list
            for j in range(min(10, len(sort_argmax))):
                ins_Id_j = sort_argmax[j] + 1

                ins_mask_j = target_ins_Ids == ins_Id_j  # size Nt

                ins_feat_j = self.target_feat[ins_mask_j, :]
                ins_feat_j_avg = torch.mean(ins_feat_j, dim=0, keepdim=True)  # size 1 x F

                # put in the list
                ins_feat_list_target[i].append(ins_feat_j_avg)

        self.centroids_source_FG = torch.squeeze(torch.stack(centroid_list_source_FG, dim=0))  # size C_fg x 1 x F -> C_fg x F
        self.ins_feat_list_target = ins_feat_list_target

        if centroids_smoothing >= 0.:
            if self.centroids_source_FG_avg is None: self.centroids_source_FG_avg = self.centroids_source_FG
            self.centroids_source_FG_avg = torch.where(self.centroids_source_FG_avg != float('inf'), self.centroids_source_FG_avg, self.centroids_source_FG)
            self.centroids_source_FG = torch.where(self.centroids_source_FG == float('inf'), self.centroids_source_FG_avg.detach(), self.centroids_source_FG)
            self.centroids_source_FG = centroids_smoothing * self.centroids_source_FG + (1 - centroids_smoothing) * self.centroids_source_FG_avg.detach()
            self.centroids_source_FG_avg = self.centroids_source_FG.detach().clone()

    def stuff_alignment(self):

        # 计算源域和目标域的背景类中心的距离
        centroids_source = self.centroids_source_BG
        centroids_target = self.centroids_target_BG

        assert centroids_source is not None and centroids_target is not None

        count, dist = 0, 0
        for i in range(len(self.BG_LABEL)):
            # Skip the centroid with inf value
            if (centroids_source[i, 0] == float('Inf')).item() == 1: continue
            if (centroids_target[i, 0] == float('Inf')).item() == 1: continue

            dist = dist + torch.mean(self.dist(centroids_source[i, :] - centroids_target[i, :]))  # size F - size F -> size 1
            count += 1

        if count == 0:
            return torch.tensor([0]).cuda()
        return dist / count

    def things_alignment(self):
        centroids_source = self.centroids_source_FG
        assert centroids_source is not None and centroids_source is not None

        i2c_count, i2c_dist = 0, 0
        i2i_count, i2i_dist = 0, 0

        for i in range(len(self.FG_LABEL)):
            if (centroids_source[i, 0] == float('Inf')).item() == 1: continue

            ins_feat_target_i = self.ins_feat_list_target[i]  # list len N_ins , item size 1 x F
            ins_feat_pool_i = self.ins_feat_pool_source[i, :, :]  # size p_capacity x F

            if len(ins_feat_target_i) != 0:
                # computer the distance from the targe instance feature  to the centroid
                ins_feat_target_i = torch.squeeze(torch.stack(self.ins_feat_list_target[i]))  # size N_ins x F
                i2c_dist = i2c_dist + self.dist(centroids_source[i, :] - ins_feat_target_i)  # size F - size N_seen x F -> size 1
                i2c_count += 1

                # computer the distance from the targe instance feature  to the source instance feature pool
                size_pool_i = self.pool_ptr[i]
                if size_pool_i != 0:
                    for ins_feat in ins_feat_target_i:
                        i2i_dist = i2i_dist + torch.min(self.dist(ins_feat_pool_i - ins_feat, dim=1))  # size F - N_ins x F ==> N_ins == > 1
                        i2i_count += 1
        if i2c_count == 0:
            i2c_dist = torch.tensor([0]).cuda()
        else:
            i2c_dist = i2c_dist / i2c_count
        if i2i_count == 0:
            i2i_dist = torch.tensor([0]).cuda()
        else:
            i2i_dist = i2i_dist / i2i_count

        return i2c_dist, i2i_dist

    def centroids_to_other(self):

        # FG
        dist_fg, count_fg = 0, 0
        centroids_source = self.centroids_source_FG
        indices = [i for i in range(self.C_fg) if (centroids_source[i, 0] == float('Inf')).item() == 0] # 只计算没有inf值的前景类
        count_fg += len(indices)
        if len(indices) == 1:
            dist_fg = torch.tensor([0]).cuda()
        else:
            for i in indices:
                indices_but_i = np.array([ind for ind in indices if ind != i])
                dist_fg = dist_fg + self.dist(centroids_source[i, :] - centroids_source[indices_but_i, :])
            if count_fg != 0:
                dist_fg /= count_fg
            else:
                dist_fg = torch.tensor([0]).cuda()

        # BG
        dist_bg, count_bg = 0, 0
        centroids_source = self.centroids_source_BG
        indices = [i for i in range(self.C_bg) if (centroids_source[i, 0] == float('Inf')).item() == 0]
        count_bg += len(indices)
        if len(indices) == 1:
            dist_bg = torch.tensor([0]).cuda()
        else:
            for i in indices:
                indices_but_i = np.array([ind for ind in indices if ind != i])
                dist_bg = dist_bg + self.dist(centroids_source[i, :] - centroids_source[indices_but_i, :])
            if count_fg != 0:
                dist_bg /= count_bg
            else:
                dist_bg = torch.tensor([0]).cuda()

        return dist_fg + dist_bg

    def similarity_dsb(self, feat_domain, temperature=1.):
        """
        Compute EM loss with the probability-based distribution of each feature
        :param feat_domain: source, target or both
        :param temperature: softmax temperature
        """

        if feat_domain == 'source':
            feat = self.source_feat  # size N x F
        elif feat_domain == 'target':
            feat = self.target_feat  # size N x F
        elif feat_domain == 'both':
            feat = torch.cat([self.source_feat, self.target_feat], dim=0)  # (Ns + Nt) x F
        else:
            raise ValueError('Wrong param used: {}    Select from: [source, target, both]'.format(feat_domain))

        ###### FG
        centroids = self.centroids_source_FG
        seen_classes = [i for i in range(self.C_fg) if not torch.isnan(centroids[i, 0]) and not centroids[i, 0] == float('Inf')]  # list of C elems, True for seen classes, False elsewhere
        centroids_filtered = centroids[seen_classes, :]  # C_seen x F

        # dot similarity between features and centroids
        z = torch.mm(feat, centroids_filtered.t())  # size N x C_seen

        # entropy loss to push each feature to be similar to only one class prototype (no supervision)
        loss_fg = -1 * torch.mean(F.softmax(z / temperature, dim=1) * F.log_softmax(z / temperature, dim=1))

        ###### BG
        centroids = self.centroids_source_BG
        seen_classes = [i for i in range(self.C_bg) if not torch.isnan(centroids[i, 0]) and not centroids[i, 0] == float('Inf')]  # list of C elems, True for seen classes, False elsewhere
        centroids_filtered = centroids[seen_classes, :]
        z = torch.mm(feat, centroids_filtered.t())
        loss_bg = -1 * torch.mean(F.softmax(z / temperature, dim=1) * F.log_softmax(z / temperature, dim=1))

        return loss_fg + loss_bg

    ## LOSSES ##
    def alignment_loss(self, alignment_params):
        norm_order = alignment_params['norm_order']
        self.dist_func = norm_order

        c2c_dist_B = self.stuff_alignment()
        i2c_dist_F, i2i_dist_F = self.things_alignment()
        c2other_dist = self.centroids_to_other()

        return c2c_dist_B, i2c_dist_F, i2i_dist_F, c2other_dist

    def orthogonality_loss(self, orthogonality_params):
        """
        Compute the feature orthogonality loss
        :param orthogonality_params:
               - temp: softmax temperature value
        """

        temp = orthogonality_params['temp']

        _, loss = self.similarity_dsb(feat_domain='both', temperature=temp)

        return loss

    def sparsity_loss(self, sparsity_params):
        """
        Compute the feature orthogonality loss
        :param sparsity_params:
               - rho: threshold value in loss
               - power: power value in sparsity loss
        """

        def loss_func(tensor, loss_type, rho, exponent=None):
            """
            :param tensor: any tensor
            :param loss_type: poly or exp
            :param rho: threshold value in loss
            :param exponent: exponent for poly type
            """
            if loss_type == 'poly':
                exp = int(exponent)
                if exponent % 2 == 0:
                    return -1 * torch.mean((tensor - rho) ** exp)
                else:
                    return -1 * torch.mean(torch.abs((tensor - rho) ** exp))
            elif loss_type == 'exp':
                return -1 * torch.abs(tensor - rho) * torch.exp(torch.abs(tensor - rho))
            else:
                raise ValueError('Loss type {} not allowed, poly or exp are the available options'.format(loss_type))

        exponent, rho = sparsity_params['norm_order'], sparsity_params['rho']

        # discard invalid centroids (those of classes still to be found)
        seen_classes = [i for i in range(self.num_class) if
                        not torch.isnan(self.mixed_centroids[i, 0]) and not self.mixed_centroids[i, 0] == float('Inf')]
        # normalize in [0,1]
        centroids_normalized = self.mixed_centroids[seen_classes, :] / torch.unsqueeze(
            torch.max(self.mixed_centroids[seen_classes, :], dim=-1)[0], dim=-1)

        loss = loss_func(centroids_normalized, loss_type='poly', rho=rho, exponent=exponent)

        return loss

    def forward(self, **kwargs):

        # 特征处理
        # 传入特征图、预测结果预测结果softmax（置信度）、GT、源域还是目标域
        # 计算self.source_feat、self.source_gt
        self.feature_processing(feat=kwargs.get('source_feat'), softmax=kwargs.get('source_prob'), gt=kwargs.get('source_gt'), domain='source')
        self.feature_processing(feat=kwargs.get('target_feat'), softmax=kwargs.get('target_prob'), gt=kwargs.get('target_gt'), domain='target')

        smo_coeff = kwargs['smo_coeff']
        assert smo_coeff <= 1., 'Centroid smoothing coefficient with invalid value: {}'.format(smo_coeff)
        self.computer_things(centroids_smoothing=smo_coeff)
        self.computer_stuff(centroids_smoothing=smo_coeff)

        # 背景：类中心到类中心的距离
        # 前景：实例到类中心的距离、实例到实例的距离
        # 其他：类中心到其他类的距离
        c2c_dist_B, i2c_dist_F, i2i_dist_F, c2other_dist = self.alignment_loss(kwargs.get('alignment_params'))

        output = {'c2c_dist_B': c2c_dist_B, 'i2c_dist_F': i2c_dist_F, 'i2i_dist_F': i2i_dist_F, 'c2other_dist': c2other_dist}
        return output


class IW_MaxSquareloss(nn.Module):
    def __init__(self, ignore_index=-1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
        self.ratio = ratio

    def forward(self, pred, prob, label=None):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :param label(optional): the map for counting label numbers (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        # prob -= 0.5
        N, C, H, W = prob.size()
        mask = (prob != self.ignore_index)
        maxpred, argpred = torch.max(prob, 1)   # 计算最大概率和对应类别序号
        mask_arg = (maxpred != self.ignore_index)   # 计算忽略类别的掩码
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long) * self.ignore_index) # 用忽略类别的掩码，将忽略的
        if label is None:
            label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(),
                               bins=self.num_class + 1, min=-1,
                               max=self.num_class - 1).float()
            hist = hist[1:]
            weight = (1 / torch.max(torch.pow(hist, self.ratio) * torch.pow(hist.sum(), 1 - self.ratio), torch.ones(1))).to(argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        prior = torch.mean(prob, (2, 3), True).detach()
        loss = -torch.sum((torch.pow(prob, 2) * weights)[mask]) / (batch_size * self.num_class)
        return loss


if __name__ == '__main__':
    # f = feat_reg_ST_loss(feat_channel=3)
    #
    # f.dist_func = 1
    torch.manual_seed(1)
    feature_s = torch.randint(0, 9, (2, 512, 512, 512), dtype=torch.float).cuda()
    feature_t = torch.randint(0, 9, (2, 512, 512, 512), dtype=torch.float).cuda()
    feat_s_softmax = torch.nn.functional.softmax(torch.rand((2, 18, 512, 512)), dim=1).cuda()
    feat_t_softmax = torch.nn.functional.softmax(torch.rand((2, 18, 512, 512)), dim=1).cuda()

    _, gt_feat_s = torch.max(feat_s_softmax, dim=1)  # size B x h x w
    softmax_s = F.interpolate(feat_s_softmax, size=(2, 2), mode='bilinear', align_corners=True)  # size B x C x h x w
    _, gt_s = torch.max(softmax_s, dim=1)  # size B x h x w

    _, gt_feat_t = torch.max(feat_t_softmax, dim=1)  # size B x h x w
    softmax_t = F.interpolate(feat_t_softmax, size=(2, 2), mode='bilinear', align_corners=True)  # size B x C x h x w
    _, gt_t = torch.max(softmax_t, dim=1)  # size B x h x w

    # f.feature_processing(feature_s, feat_s_softmax, gt_s, domain='source')
    # f.feature_processing(feature_t, feat_t_softmax, gt_t, domain='target')
    #
    # f.computer_stuff(centroids_smoothing=0.9)
    # f.computer_things(centroids_smoothing=0.9)
    #
    # i2c_dist_F, i2i_dist_F = f.things_alignment()
    # c2c_dist_B = f.stuff_alignment()
    # c2other_c_dist = f.centroids_to_other()
    f = feat_reg_ST_loss(feat_channel=512)
    loss_kwargs = {}
    alignment_params = {'norm_order': 1}
    loss_kwargs['alignment_params'] = alignment_params

    loss_kwargs['source_prob'] = feat_s_softmax
    loss_kwargs['target_prob'] = feat_t_softmax
    loss_kwargs['source_feat'] = feature_s
    loss_kwargs['target_feat'] = feature_t
    loss_kwargs['source_gt'] = gt_s
    loss_kwargs['target_gt'] = gt_t
    loss_kwargs['smo_coeff'] = 0.9
    loss = f(**loss_kwargs)
    print(loss)
