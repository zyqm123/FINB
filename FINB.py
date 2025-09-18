import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from model.registry import FINETUNING
from utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionMatching(nn.Module):
    def __init__(self, in_channels=3):
        super(AttentionMatching, self).__init__()
        self.conv_spt = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
        )
        self.conv_qry = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1),
        )

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        self.sigmoid = nn.Sigmoid()

    def correlation_matrix(self, sou_fts, tar_fts):
        sou_norm = F.normalize(sou_fts, p=2, dim=1)
        tar_norm = F.normalize(tar_fts, p=2, dim=1)
        similarity = torch.sum(sou_norm * tar_norm, dim=1, keepdim=True)
        return similarity

    def forward(self, sou_fts, tar_fts, band):
        sou_proj = F.relu(self.conv_spt(sou_fts))
        tar_proj = F.relu(self.conv_qry(tar_fts))

        similarity_matrix = self.sigmoid(self.correlation_matrix(sou_fts, tar_fts))

        if band in ['low', 'high']:
            weighted_sou = (1 - similarity_matrix) * sou_proj
            weighted_tar = (1 - similarity_matrix) * tar_proj
        else:
            weighted_sou = similarity_matrix * sou_proj
            weighted_tar = similarity_matrix * tar_proj

        combined = torch.cat([weighted_sou, weighted_tar], dim=1)
        fused_tensor = F.relu(self.conv_fusion(combined))

        return fused_tensor


class FAM(nn.Module):
    def __init__(self, feature_dim, in_channels):
        super(FAM, self).__init__()
        self.feature_dim = feature_dim
        self.N = feature_dim * feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapt_pooling = nn.AdaptiveAvgPool2d((feature_dim, feature_dim))
        self.attention_matching = ChannelEnhancedAttention(in_channels=in_channels).to(device=self.device)

    def forward(self, sou_fts, tar_fts):
        sou_fts = self.adapt_pooling(sou_fts)
        tar_fts = self.adapt_pooling(tar_fts)

        sou_low, sou_mid, sou_high = self.filter_tensor(sou_fts)
        tar_low, tar_mid, tar_high = self.filter_tensor(tar_fts)

        fused_low = self.attention_matching(sou_low, tar_low, 'low')
        fused_mid = self.attention_matching(sou_mid, tar_mid, 'mid')
        fused_high = self.attention_matching(sou_high, tar_high, 'high')

        return fused_low, fused_mid, fused_high

    def filter_tensor(self, tensor, cutoff=0.3):
        B, C, H, W = tensor.shape
        tensor = tensor.float()

        fft_tensor = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))

        max_radius = np.sqrt((H // 2) ** 2 + (W // 2) ** 2)
        low_cutoff = max_radius * cutoff
        high_cutoff = max_radius * (1 - cutoff)

        y, x = torch.meshgrid(torch.arange(H, device=self.device),
                              torch.arange(W, device=self.device), indexing='ij')
        center_y, center_x = H // 2, W // 2
        distance = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

        low_mask = (distance <= low_cutoff).float()[None, None, :, :]
        high_mask = (distance >= high_cutoff).float()[None, None, :, :]
        mid_mask = ((distance > low_cutoff) & (distance < high_cutoff)).float()[None, None, :, :]

        low_fft = fft_tensor * low_mask
        mid_fft = fft_tensor * mid_mask
        high_fft = fft_tensor * high_mask

        low_tensor = torch.fft.ifft2(torch.fft.ifftshift(low_fft, dim=(-2, -1)), dim=(-2, -1)).real
        mid_tensor = torch.fft.ifft2(torch.fft.ifftshift(mid_fft, dim=(-2, -1)), dim=(-2, -1)).real
        high_tensor = torch.fft.ifft2(torch.fft.ifftshift(high_fft, dim=(-2, -1)), dim=(-2, -1)).real
        return low_tensor, mid_tensor, high_tensor


class ChannelEnhancedAttention(AttentionMatching):
    def __init__(self, in_channels=3):
        super().__init__(in_channels * 2)
        self.res_conv = nn.Conv2d(in_channels * 2, in_channels * 2, 1)
        self.channel_reduction = nn.Conv2d(in_channels * 2, in_channels, 1)

    def forward(self, x1, x2, band):
        x1 = torch.cat([x1, x1.mean(dim=1, keepdim=True).expand_as(x1)], dim=1)
        x2 = torch.cat([x2, x2.std(dim=1, keepdim=True).expand_as(x2)], dim=1)
        tmp = super().forward(x1, x2, band)
        a = self.res_conv(x1)
        tmp = tmp + a
        output = self.channel_reduction(tmp)
        return output


class MultiScaleFAM(nn.Module):
    def __init__(self, feature_dim, in_channels):
        super().__init__()
        self.scales = [1, 2, 4]
        self.fams = nn.ModuleList(
            [FAM(feature_dim // s, in_channels=in_channels) for s in self.scales]
        )

    def forward(self, x1, x2):
        outputs = []
        for s, fam in zip(self.scales, self.fams):
            x1_resized = F.avg_pool2d(x1, s) if s > 1 else x1
            x2_resized = F.avg_pool2d(x2, s) if s > 1 else x2
            low, mid, high = fam(x1_resized, x2_resized)
            low_up = F.interpolate(low, scale_factor=s)
            mid_up = F.interpolate(mid, scale_factor=s)
            high_up = F.interpolate(high, scale_factor=s)
            outputs.append(low_up + mid_up + high_up)
        return sum(outputs) / len(outputs)


class EnhancedInterDomainNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.FAM_low = MultiScaleFAM(config.transformer.image_size // 4, in_channels=3)
        self.FAM_high = MultiScaleFAM(config.transformer.image_size, in_channels=3)

        self.domain_disc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Flatten(0)
        )
        self.encoder = self.encoder.to(device)
        self.FAM_low = self.FAM_low.to(device)
        self.FAM_high = self.FAM_high.to(device)

    def contrastive_shared_loss(self, fuse_feat, src_feat, tgt_feat, temperature=0.1):
        pos_sim = F.cosine_similarity(fuse_feat, src_feat).mean() + \
                  F.cosine_similarity(fuse_feat, tgt_feat).mean()
        neg_sim = F.cosine_similarity(src_feat, tgt_feat).mean()
        return -pos_sim + 0.5 * neg_sim

    def forward(self, src, tgt):
        way, shot, C, H, W = src.shape
        src = src.view(way * shot, C, H, W)
        tgt = tgt.view(way * shot, C, H, W)

        inter_low = self.FAM_low(F.avg_pool2d(src, 4), F.avg_pool2d(tgt, 4))
        inter_high = self.FAM_high(src, tgt)

        fuse_feat = inter_high + F.interpolate(inter_low, scale_factor=4)
        fuse_feat = fuse_feat.view(way * shot, C, H, W)
        domain_loss = self.contrastive_shared_loss(fuse_feat, src, tgt)

        return fuse_feat, domain_loss


@FINETUNING.register
class FINBFinetuning():

    def __init__(self, config, model):
        self.model = model
        self.model_inter = copy.deepcopy(model)
        self.model_source = copy.deepcopy(model)

        self.way = config.way
        self.shot = config.shot
        self.query_shot = config.query_shot
        self.criterion = nn.CrossEntropyLoss()

        self.Inter = EnhancedInterDomainNet(config)

    def meta_finetuning(self, data, sou_data, config):
        epoch = config.epoch
        m_inter = config.m_inter
        m_target = config.m_target
        support = data[:self.way * self.shot]
        sou_support = sou_data[:self.way * self.shot]

        self.prototypical_feature_reprojection1(support, sou_support, epoch, m_inter, m_target)
        self.Instance_level_feature_recalibration(support, data)
        self.prototypical_feature_reprojection2(support, epoch)

    def prototypical_feature_reprojection1(self, support, sou_support, epoch, m_inter, m_target):
        support_feat = self.model.get_feature_map(support).view(self.way, self.shot, self.model.resolution,
                                                                self.model.d)
        support_view = support.view(self.way, self.shot, *support.shape[-3:])
        sou_support_view = sou_support.view(self.way, self.shot, *sou_support.shape[-3:])

        for t in range(epoch):
            inter_support, domain_loss = self.Inter(support_view, sou_support_view)
            inter_support_feat = self.model_inter.get_feature_map(inter_support).view(self.way, self.shot,
                                                                                      self.model.resolution,
                                                                                      self.model.d)
            inter_support_view = inter_support.view(self.way, self.shot, *inter_support.shape[-3:])

            indexes = torch.randperm(self.shot)
            s_idxes = indexes[:1]
            if self.shot > 1:
                q_idxes = indexes[1:]
            else:
                q_idxes = indexes[:1]

            if self.shot > 1:
                inter_part_quft = inter_support_view[:, q_idxes].view(self.way * (self.shot - 1),
                                                                      *inter_support.shape[-3:])
                part_quft = support_view[:, q_idxes].view(self.way * (self.shot - 1), *support.shape[-3:])
            else:
                inter_part_quft = inter_support_view[:, q_idxes].view(self.way * (self.shot), *inter_support.shape[-3:])
                part_quft = support_view[:, q_idxes].view(self.way * (self.shot), *support.shape[-3:])

            inter_part_spft = inter_support_feat[:, s_idxes].squeeze(1)
            self.model_inter.cat_mat = nn.Parameter(inter_part_spft)
            part_spft = support_feat[:, s_idxes].squeeze(1)
            self.model.cat_mat = nn.Parameter(part_spft)

            if self.shot > 1:
                self.step(self.shot - 1, inter_part_quft, part_quft, domain_loss)
            else:
                self.step(self.shot, inter_part_quft, part_quft, domain_loss)

            with torch.no_grad():
                for param_s, param_i, param_t in zip(self.model_source.parameters(), self.model_inter.parameters(),
                                                     self.model.parameters()):
                    param_i.data = param_i.data * m_inter + param_s.data * (1. - m_inter)
                    param_t.data = param_t.data * m_target + param_i.data * (1. - m_target)

    def prototypical_feature_reprojection2(self, support, epoch):
        optimizer_2nd = torch.optim.SGD([self.model.cat_mat], lr=0.01, momentum=0.9, weight_decay=0.001, nesterov=True)
        for t in range(epoch):
            self.step1(self.shot, support, optimizer_2nd)

    def Instance_level_feature_recalibration(self, support, data):
        weight = torch.zeros(self.way, self.shot).cuda()
        with torch.no_grad():
            support_feat = self.model.get_feature_map(support).view(self.way, self.shot, self.model.resolution,
                                                                    self.model.d)
            data_feat = self.model.get_feature_map(data).reshape(self.way, self.shot + self.query_shot,
                                                                 self.model.resolution, self.model.d)
            for cls in range(self.way):
                dis = self.compute_d(support_feat[cls], data_feat[cls]).mean(1)
                dis = 1 / dis
                weight[cls] += (dis / dis.sum())
        weighted_support_feat = support_feat * weight[:, :, None, None]
        self.model.cat_mat = nn.Parameter(weighted_support_feat.mean(1))

    def compute_d(self, f1, f2):
        t1 = f1.unsqueeze(1).expand(len(f1), len(f2), f1.shape[1], f1.shape[2])
        t2 = f2.unsqueeze(0).expand(len(f1), len(f2), f2.shape[1], f2.shape[2])
        d = (t1 - t2).pow(2).sum((2, 3))
        return d

    def step(self, q_shot, query_inter, query, domain_loss):
        target = torch.LongTensor([i // q_shot for i in range(q_shot * self.way)]).cuda()
        optimizer = torch.optim.SGD(list(self.model.feature_extractor.parameters()) + list(
            self.model_inter.feature_extractor.parameters()) + list(self.Inter.parameters()), lr=0.01, momentum=0.9,
                                    weight_decay=5e-4, nesterov=True)

        outputs_inter = self.model_inter(query_inter)
        loss_inter = self.criterion(outputs_inter, target)

        outputs = self.model(query)
        loss_target = self.criterion(outputs, target)

        loss = loss_inter + loss_target + domain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def step1(self, q_shot, query, optimizer):
        target = torch.LongTensor([i // q_shot for i in range(q_shot * self.way)]).cuda()
        outputs = self.model(query)
        loss = self.criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def set_cm(self, support):
        support_feat = self.model.get_feature_map(support).view(self.way, self.shot, self.model.resolution,
                                                                self.model.d)
        self.model.cat_mat = nn.Parameter(support_feat.mean(1))