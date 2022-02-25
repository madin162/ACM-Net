import torch 
import torch.nn as nn 
from itertools import compress
import numpy as np
from scipy import ndimage
from itertools import combinations
from torch.autograd import Function
import math

beta = [-2, -2]

class GradRevLayer(Function):
    @staticmethod
    def forward(ctx, x, beta):
        ctx.beta = beta
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.beta
        return grad_input, None


class ACMNet(nn.Module):
    
    def __init__(self, args):
        super(ACMNet, self).__init__()
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim
        self.action_cls_num = args.action_cls_num # Only the action categories number.
        self.drop_thresh = args.dropout
        self.ins_topk_seg = args.ins_topk_seg 
        self.con_topk_seg = args.con_topk_seg 
        self.bak_topk_seg = args.bak_topk_seg
        
        self.dropout = nn.Dropout(args.dropout)
        if self.dataset == "THUMOS":
            self.feature_embedding = nn.Sequential(
                # nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            self.feature_embedding = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        
        # We introduce three-branch attention, action instance, action context and the irrelevant backgrounds.
        self.att_branch = nn.Conv1d(in_channels=self.feature_dim, out_channels=3, kernel_size=1, padding=0)
        self.snippet_cls = nn.Linear(in_features=self.feature_dim, out_features=(self.action_cls_num + 1))
        
    def forward(self, input_features):

        device = input_features.device
        batch_size, temp_len = input_features.shape[0], input_features.shape[1]
        
        inst_topk_num = max(temp_len // self.ins_topk_seg, 1)
        cont_topk_num = max(temp_len // self.con_topk_seg, 1)
        back_topk_num = max(temp_len // self.bak_topk_seg, 1)
        
        input_features = input_features.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_features)
        
        if self.dataset == "THUMOS":
            temp_att = self.att_branch((embeded_feature))
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            temp_att = self.att_branch(self.dropout(embeded_feature))
        
        temp_att = temp_att.permute(0, 2, 1)
        temp_att = torch.softmax(temp_att, dim=2)
        
        act_inst_att = temp_att[:, :, 0].unsqueeze(2)
        act_cont_att = temp_att[:, :, 1].unsqueeze(2)
        act_back_att = temp_att[:, :, 2].unsqueeze(2)

        embeded_feature = embeded_feature.permute(0, 2, 1)
        embeded_feature_rev = embeded_feature
        
        select_idx = torch.ones((batch_size, temp_len, 1), device=device)
        select_idx = self.dropout(select_idx)
        embeded_feature = embeded_feature * select_idx

        act_cas = self.snippet_cls(self.dropout(embeded_feature))
        act_inst_cas = act_cas * act_inst_att
        act_cont_cas = act_cas * act_cont_att
        act_back_cas = act_cas * act_back_att
        
        sorted_inst_cas, _ = torch.sort(act_inst_cas, dim=1, descending=True)
        sorted_cont_cas, _ = torch.sort(act_cont_cas, dim=1, descending=True)
        sorted_back_cas, _ = torch.sort(act_back_cas, dim=1, descending=True)
        
        act_inst_cls = torch.mean(sorted_inst_cas[:, :inst_topk_num, :], dim=1)
        act_cont_cls = torch.mean(sorted_cont_cas[:, :cont_topk_num, :], dim=1)
        act_back_cls = torch.mean(sorted_back_cas[:, :back_topk_num, :], dim=1)
        act_inst_cls = torch.softmax(act_inst_cls, dim=1)
        act_cont_cls = torch.softmax(act_cont_cls, dim=1)
        act_back_cls = torch.softmax(act_back_cls, dim=1)
        
        act_inst_cas = torch.softmax(act_inst_cas, dim=2)
        act_cont_cas = torch.softmax(act_cont_cas, dim=2)
        act_back_cas = torch.softmax(act_back_cas, dim=2)
        
        act_cas = torch.softmax(act_cas, dim=2)
        
        _, sorted_act_inst_att_idx = torch.sort(act_inst_att, dim=1, descending=True)
        _, sorted_act_cont_att_idx = torch.sort(act_cont_att, dim=1, descending=True)
        _, sorted_act_back_att_idx = torch.sort(act_back_att, dim=1, descending=True)
        act_inst_feat_idx = sorted_act_inst_att_idx[:, :inst_topk_num, :].expand([-1, -1, self.feature_dim])
        act_cont_feat_idx = sorted_act_cont_att_idx[:, :cont_topk_num, :].expand([-1, -1, self.feature_dim])
        act_back_feat_idx = sorted_act_back_att_idx[:, :back_topk_num, :].expand([-1, -1, self.feature_dim])
        act_inst_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_inst_feat_idx), dim=1)
        act_cont_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_cont_feat_idx), dim=1)
        
        #background for mining
        act_back_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_back_feat_idx), dim=1)
        
        return act_inst_cls, act_cont_cls, act_back_cls,\
               act_inst_feat, act_cont_feat, act_back_feat,\
               temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas


class AdvDomainClsBase(nn.Module):
    def __init__(self, in_feat, hidden_size, type_adv, args):
        super(AdvDomainClsBase, self).__init__()
        # ====== collect arguments ====== #
        self.num_f_maps = in_feat

        self.fc1 = nn.Linear(in_feat, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.type_adv='frames'

    def forward(self, input_data, beta):
        feat = GradRevLayer.apply(input_data, beta)
        if self.type_adv == "video" and self.DA_adv_video == "rev_grad_ssl_2":
            num_seg = int(input_data.size(-1) / self.num_f_maps)
            feat = feat.reshape(
                -1, num_seg, self.num_f_maps
            )  # reshape --> (video#, seg#, dim)

            # get the pair indices
            id_pair = torch.tensor(
                list(combinations(range(num_seg), 2))
            ).long()  # all possible indices
            if self.pair_ssl == "adjacent":
                id_pair = torch.tensor([(i, i + 1)
                                        for i in range(num_seg - 1)])
            if input_data.get_device() >= 0:
                id_pair = id_pair.to(input_data.get_device())

            # get the pairwise features
            feat = feat[:, id_pair, :]  # (video#, pair#, 2, dim)
            # (video# x pair#, 2 x dim)
            feat = feat.reshape(-1, self.num_f_maps * 2)
            feat = self.fc_pair(feat)  # (video# x pair#, dim)
            feat = feat.reshape(
                -1, id_pair.size(0) * self.num_f_maps
            )  # (video#, pair# x dim)

        feat = self.fc1(feat)
        feat = self.relu(feat)
        feat = self.dropout(feat)

        return feat

class ACMNet_da(nn.Module):
    def __init__(self, args):
        super(ACMNet_da, self).__init__()
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim
        self.action_cls_num = args.action_cls_num # Only the action categories number.
        self.drop_thresh = args.dropout
        self.ins_topk_seg = args.ins_topk_seg 
        self.con_topk_seg = args.con_topk_seg 
        self.bak_topk_seg = args.bak_topk_seg
        self.r_easy = args.r_easy
        self.r_hard = args.r_hard
        self.m = args.m
        self.M = args.M

        self.dropout = nn.Dropout(args.dropout)
        if self.dataset == "THUMOS":
            self.feature_embedding = nn.Sequential(
                # nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            self.feature_embedding = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        
        # We introduce three-branch attention, action instance, action context and the irrelevant backgrounds.
        self.att_branch = nn.Conv1d(in_channels=self.feature_dim, out_channels=3, kernel_size=1, padding=0)
        self.snippet_cls = nn.Linear(in_features=self.feature_dim, out_features=(self.action_cls_num + 1))

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        num_f_maps = 2048
        self.ad_net_base = nn.ModuleList()
        self.ad_net_base += [AdvDomainClsBase(num_f_maps,num_f_maps, "frame",args)]
        self.ad_net_cls = nn.ModuleList()
        self.ad_net_cls += [nn.Linear(num_f_maps, 2)]

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def forward_cas_map(self, input_features):
        device = input_features.device
        batch_size, temp_len = input_features.shape[0], input_features.shape[1]

        inst_topk_num = max(temp_len // self.ins_topk_seg, 1)
        cont_topk_num = max(temp_len // self.con_topk_seg, 1)
        back_topk_num = max(temp_len // self.bak_topk_seg, 1)

        input_features = input_features.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_features)
        temp_att = self.att_branch(self.dropout(embeded_feature))
        temp_att = temp_att.permute(0, 2, 1)
        #batch_size, temp_len = temp_att.shape[0], temp_att.shape[1]

        temp_att = torch.softmax(temp_att, dim=2)
        act_inst_att = temp_att[:, :, 0].unsqueeze(2)
        act_cont_att = temp_att[:, :, 1].unsqueeze(2)
        act_back_att = temp_att[:, :, 2].unsqueeze(2)

        embeded_feature = embeded_feature.permute(0, 2, 1)
        embeded_feature_rev = embeded_feature
        
        select_idx = torch.ones((batch_size, temp_len, 1), device=device)
        select_idx = self.dropout(select_idx)
        embeded_feature = embeded_feature * select_idx

        act_cas = self.snippet_cls(self.dropout(embeded_feature))
        act_inst_cas = act_cas * act_inst_att
        act_cont_cas = act_cas * act_cont_att
        act_back_cas = act_cas * act_back_att
        
        sorted_inst_cas, _ = torch.sort(act_inst_cas, dim=1, descending=True)
        sorted_cont_cas, _ = torch.sort(act_cont_cas, dim=1, descending=True)
        sorted_back_cas, _ = torch.sort(act_back_cas, dim=1, descending=True)

        act_inst_cls = torch.mean(sorted_inst_cas[:, :inst_topk_num, :], dim=1)
        act_cont_cls = torch.mean(sorted_cont_cas[:, :cont_topk_num, :], dim=1)
        act_back_cls = torch.mean(sorted_back_cas[:, :back_topk_num, :], dim=1)
        act_inst_cls = torch.softmax(act_inst_cls, dim=1)
        act_cont_cls = torch.softmax(act_cont_cls, dim=1)
        act_back_cls = torch.softmax(act_back_cls, dim=1)
        
        act_inst_cas = torch.softmax(act_inst_cas, dim=2)
        act_cont_cas = torch.softmax(act_cont_cas, dim=2)
        act_back_cas = torch.softmax(act_back_cas, dim=2)
        
        act_cas = torch.softmax(act_cas, dim=2)
        
        _, sorted_act_inst_att_idx = torch.sort(act_inst_att, dim=1, descending=True)
        _, sorted_act_cont_att_idx = torch.sort(act_cont_att, dim=1, descending=True)
        _, sorted_act_back_att_idx = torch.sort(act_back_att, dim=1, descending=True)
        act_inst_feat_idx = sorted_act_inst_att_idx[:, :inst_topk_num, :].expand([-1, -1, self.feature_dim])
        act_cont_feat_idx = sorted_act_cont_att_idx[:, :cont_topk_num, :].expand([-1, -1, self.feature_dim])
        act_back_feat_idx = sorted_act_back_att_idx[:, :back_topk_num, :].expand([-1, -1, self.feature_dim])
        act_inst_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_inst_feat_idx), dim=1)
        act_cont_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_cont_feat_idx), dim=1)
        act_back_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_back_feat_idx), dim=1)
        
        return act_inst_cls, act_cont_cls, act_back_cls,\
               act_inst_feat, act_cont_feat, act_back_feat,\
               temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas
    
    def hard_snippets_mining(self, actionness, embeddings, k_hard):
        aness_np = actionness.cpu().detach().numpy() #np => numpy; aness = actionness
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def actionness_module(self,input_features):
        device = input_features.device
        batch_size, temp_len = input_features.shape[0], input_features.shape[1]

        input_features = input_features.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_features)
        temp_att = self.att_branch(self.dropout(embeded_feature))
        temp_att = temp_att.permute(0, 2, 1)
        #batch_size, temp_len = temp_att.shape[0], temp_att.shape[1]

        temp_att = torch.softmax(temp_att, dim=2)
        act_inst_att = temp_att[:, :, 0].unsqueeze(2)
        act_cont_att = temp_att[:, :, 1].unsqueeze(2)
        act_back_att = temp_att[:, :, 2].unsqueeze(2)

        embeded_feature = embeded_feature.permute(0, 2, 1)
        embeded_feature_rev = embeded_feature
        
        select_idx = torch.ones((batch_size, temp_len, 1), device=device)
        select_idx = self.dropout(select_idx)
        embeded_feature = embeded_feature * select_idx

        act_cas = self.snippet_cls(self.dropout(embeded_feature))

        actionness = act_cas.sum(dim=2)

        return embeded_feature, act_cas,actionness

    def create_contrast_pairs(self,tgt_input_features,src_bkg_feat,src_act_feat):
        #action snippets mining
        embeddings, cas, actionness = self.actionness_module(tgt_input_features)

        num_segments = tgt_input_features.shape[1]
        k_easy = num_segments // self.r_easy
        k_hard = num_segments // self.r_hard
        hard_act, hard_bkg = self.hard_snippets_mining(actionness, embeddings, k_hard)
        device = hard_act.device
        num_batch = tgt_input_features.shape[0]
        
        # get easy action and background
        df = torch.reshape(src_bkg_feat,(np.prod(np.array(src_bkg_feat.shape[0:2])),2048))
        #print(df.shape)
        src_bkg_feat = df[torch.nonzero(df.abs().sum(axis=1)).squeeze(),:]
        src_bkg_feat = src_bkg_feat[np.random.choice(src_bkg_feat.shape[0],num_batch*k_easy,replace=True),:]
        src_bkg_feat = torch.reshape(src_bkg_feat,[num_batch,k_easy,src_bkg_feat.shape[1]])
        easy_bkg = src_bkg_feat.to(device)
        #easy_bkg = torch.nn.functional.softmax(easy_bkg)
        
        
        df = torch.reshape(src_act_feat,(np.prod(np.array(src_act_feat.shape[0:2])),2048))
        #print(df.shape)
        src_act_feat = df[torch.nonzero(df.abs().sum(axis=1)).squeeze(),:]
        src_act_feat = src_act_feat[np.random.choice(src_act_feat.shape[0],num_batch*k_easy,replace=True),:]
        src_act_feat = torch.reshape(src_act_feat,[num_batch,k_easy,src_act_feat.shape[1]])
        easy_act = src_act_feat.to(device)
        #easy_act = torch.nn.functional.softmax(easy_act)

        #print(hard_act.min())
        #print(hard_act.max())
        
        #print(easy_act.min())
        #print(easy_act.max())


        #background mining
        contrast_pairs = {
            'EA': easy_act, #for act: x+
            'EB': easy_bkg, #for act: x-
            'HA': hard_act, #x
            'HB': hard_bkg  #x
        }
        return contrast_pairs

    def forward_dom_pred(self, feature_source,feature_target,reverse):

        pred_d_source, label_d_source = self.forward_domain(feature_source, 0, beta, reverse)
        pred_d_target, label_d_target = self.forward_domain(feature_target, 1, beta, reverse)

        # concatenate domain predictions & labels (frame-level)
        # Local SSTDA
        pred_d = torch.cat((pred_d_source, pred_d_target), 0)
        label_d = torch.cat((label_d_source, label_d_target), 0).long()

        # return iou_map, regu_s, regu_e
        return (pred_d,label_d)

    def predict_domain_frame(self, feat, beta_value):
        dim_feat = feat.size(2)
        num_frame = feat.size(1)
        feat = feat.transpose(1, 2).reshape(-1, dim_feat)  # reshape to (batch x frame#, dim) [400, 256]
        out = self.ad_net_base[0](feat, beta_value)
        out = self.ad_net_cls[0](out)  # (batch x frame#, 2)
        out = out.reshape(-1, num_frame, 2).transpose(1, 2)  # reshape back to (batch, 2, frame#)
        out = out.unsqueeze(1)  # (batch, 1, 2, frame#)

        return out

    def forward_domain(self, x, domain_GT, beta, reverse):
        if reverse:  # reverse the gradient
            x = GradRevLayer.apply(x, beta[0])
        
        # compute domain predictions for single stage
        out_d, lb_d = self.forward_stage(x, beta, domain_GT)

        return out_d,lb_d

    def forward_strong_sup(self, input_features):
        device = input_features.device
        batch_size, temp_len = input_features.shape[0], input_features.shape[1]

        inst_topk_num = max(temp_len // self.ins_topk_seg, 1)
        cont_topk_num = max(temp_len // self.con_topk_seg, 1)
        back_topk_num = max(temp_len // self.bak_topk_seg, 1)

        input_features = input_features.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_features)
        
        #(act_inst_cls, act_cont_cls, act_back_cls,\
        #act_inst_feat, act_cont_feat, act_back_feat,\
        #temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas)=self.forward_cas_map(self, embeded_feature)

        start = self.x_1d_s(embeded_feature)
        end = self.x_1d_e(embeded_feature)
        return start, end

    def forward_stage(self, out_feat, beta, domain_GT):
        out_d = self.predict_domain_frame(
            out_feat, beta[0]
        )
        
        #out_feat_video = self.aggregate_frames(out_d, mask)


        # === Select valid frames + Generate domain labels === #
        out_d, lb_d = self.select_masked(out_d, domain_GT)
        #print(out_d.size())

        return out_d, lb_d
    
    def aggregate_frames(self, out_feat, mask):
        dim_feat = out_feat.size(1)
        num_batch = out_feat.size(0)

        # calculate total frame # for each video
        num_total_frame = mask[:, 0, :].sum(-1)

        # make sure the total frame# can be divided by seg#
        num_frame_seg = (num_total_frame / self.num_seg).int()
        num_frame_new = self.num_seg * num_frame_seg

        # reshape frame-level features based on num_seg --> aggregate frames
        out_feat_video_batch = out_feat[0, :, : num_frame_new[0]].reshape(
            dim_feat, self.num_seg, num_frame_seg[0]
        )  # (dim, seg#, seg_frame#)
        out_feat_video_batch = (
            out_feat_video_batch.sum(-1) / num_frame_seg[0]
        )  # average all the features in a segment ==> (dim, seg#)
        out_feat_video = out_feat_video_batch.unsqueeze(0)  # (1, dim, seg#)
        for b in range(1, num_batch):
            out_feat_video_batch = out_feat[b, :, : num_frame_new[b]].reshape(
                dim_feat, self.num_seg, num_frame_seg[b]
            )
            out_feat_video_batch = out_feat_video_batch.sum(-1) / (
                num_frame_seg[b].float()
            )
            out_feat_video = torch.cat(
                (out_feat_video, out_feat_video_batch.unsqueeze(0)), dim=0
            )  # (batch, dim, seg#)

        return out_feat_video

    def select_masked(self, out_d, domain_GT):
        num_class_domain = out_d.size(1)
        out_d = (out_d.transpose(2, 3).transpose(1, 2).reshape(-1, num_class_domain, 2))  # (batch x frame#, class#, 2)
        lb_d = torch.full_like(out_d[:, :, 0], domain_GT)
        return out_d, lb_d

    def forward(self, src_input_feature,tgt_input_feature,src_bkg_feat,src_act_feat,infer=0):

        if(infer==0):
            # === Weak-supervised learning on target domain ===
            tgt_act_inst_cls, tgt_act_cont_cls, tgt_act_back_cls,\
            tgt_act_inst_feat, tgt_act_cont_feat, tgt_act_back_feat,\
            tgt_temp_att, tgt_act_inst_cas, _, _, _=self.forward_cas_map(tgt_input_feature)


            # weakly supervised learning
            src_act_inst_cls, src_act_cont_cls, src_act_back_cls,\
            src_act_inst_feat, src_act_cont_feat, src_act_back_feat,\
            src_temp_att, src_act_inst_cas, src_act_cas, _, _ =self.forward_cas_map(src_input_feature)

            # Strong supervision learning
            src_start, src_end = self.forward_strong_sup(src_input_feature)

            # Domain adaptation
            # Background alignment
            contrast_pairs = self.create_contrast_pairs(tgt_input_feature,src_bkg_feat,src_act_feat)

            # adversarial domain learning
            pred_d, label_d = self.forward_dom_pred(src_input_feature,tgt_input_feature, reverse=False)
            

            # return 
            return  tgt_act_inst_cls,tgt_act_cont_cls, tgt_act_back_cls,tgt_act_inst_feat,\
            tgt_act_cont_feat, tgt_act_back_feat,tgt_temp_att, tgt_act_inst_cas,\
            src_act_inst_cls, src_act_cont_cls, src_act_back_cls,\
            src_act_inst_feat, src_act_cont_feat, src_act_back_feat,\
            src_temp_att, src_act_inst_cas, src_act_cas,\
            src_start, src_end, contrast_pairs, pred_d, label_d
        else:
            act_inst_cls, act_cont_cls, act_back_cls,\
            act_inst_feat, act_cont_feat, act_back_feat,\
            temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas = self.forward_cas_map(src_input_feature)

            return act_inst_cls, act_cont_cls, act_back_cls,\
            act_inst_feat, act_cont_feat, act_back_feat,\
            temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas


class ACMNet_src(nn.Module):
    def __init__(self, args):
        super(ACMNet_src, self).__init__()
        self.dataset = args.dataset
        self.feature_dim = args.feature_dim
        self.action_cls_num = args.action_cls_num # Only the action categories number.
        self.drop_thresh = args.dropout
        self.ins_topk_seg = args.ins_topk_seg 
        self.con_topk_seg = args.con_topk_seg 
        self.bak_topk_seg = args.bak_topk_seg
        self.r_easy = args.r_easy
        self.r_hard = args.r_hard
        self.m = args.m
        self.M = args.M
        self.tscale = args.sample_segments_num
        self.prop_boundary_ratio = args.prop_boundary_ratio
        self.num_sample = args.num_sample
        self.num_sample_perbin = args.num_sample_perbin

        self.dropout = nn.Dropout(args.dropout)

        self.hidden_dim_1d = self.feature_dim
        self.hidden_dim_2d = 128
        self.hidden_dim_3d = 512

        self._get_interp1d_mask()

        if self.dataset == "THUMOS":
            self.feature_embedding = nn.Sequential(
                # nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            self.feature_embedding = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        
        # We introduce three-branch attention, action instance, action context and the irrelevant backgrounds.
        self.att_branch = nn.Conv1d(in_channels=self.feature_dim, out_channels=3, kernel_size=1, padding=0)
        self.snippet_cls = nn.Linear(in_features=self.feature_dim, out_features=(self.action_cls_num + 1))

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        #self.x_1d_p = nn.Sequential(
        #    nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
        #    nn.ReLU(inplace=True)
        #)
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )

        #Domain adaptation
        num_f_maps = 2048
        self.ad_net_base = nn.ModuleList()
        self.ad_net_base += [AdvDomainClsBase(num_f_maps,num_f_maps, "frame",args)]
        self.ad_net_cls = nn.ModuleList()
        self.ad_net_cls += [nn.Linear(num_f_maps, 2)]

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def forward_cas_map(self, input_features):
        device = input_features.device
        batch_size, temp_len = input_features.shape[0], input_features.shape[1]

        inst_topk_num = max(temp_len // self.ins_topk_seg, 1)
        cont_topk_num = max(temp_len // self.con_topk_seg, 1)
        back_topk_num = max(temp_len // self.bak_topk_seg, 1)

        input_features = input_features.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_features)
        temp_att = self.att_branch(self.dropout(embeded_feature))
        temp_att = temp_att.permute(0, 2, 1)
        #batch_size, temp_len = temp_att.shape[0], temp_att.shape[1]

        temp_att = torch.softmax(temp_att, dim=2)
        act_inst_att = temp_att[:, :, 0].unsqueeze(2)
        act_cont_att = temp_att[:, :, 1].unsqueeze(2)
        act_back_att = temp_att[:, :, 2].unsqueeze(2)

        embeded_feature = embeded_feature.permute(0, 2, 1)
        embeded_feature_rev = embeded_feature
        
        select_idx = torch.ones((batch_size, temp_len, 1), device=device)
        select_idx = self.dropout(select_idx)
        embeded_feature = embeded_feature * select_idx

        act_cas = self.snippet_cls(self.dropout(embeded_feature))
        act_inst_cas = act_cas * act_inst_att
        act_cont_cas = act_cas * act_cont_att
        act_back_cas = act_cas * act_back_att
        
        sorted_inst_cas, _ = torch.sort(act_inst_cas, dim=1, descending=True)
        sorted_cont_cas, _ = torch.sort(act_cont_cas, dim=1, descending=True)
        sorted_back_cas, _ = torch.sort(act_back_cas, dim=1, descending=True)

        act_inst_cls = torch.mean(sorted_inst_cas[:, :inst_topk_num, :], dim=1)
        act_cont_cls = torch.mean(sorted_cont_cas[:, :cont_topk_num, :], dim=1)
        act_back_cls = torch.mean(sorted_back_cas[:, :back_topk_num, :], dim=1)
        act_inst_cls = torch.softmax(act_inst_cls, dim=1)
        act_cont_cls = torch.softmax(act_cont_cls, dim=1)
        act_back_cls = torch.softmax(act_back_cls, dim=1)
        
        act_inst_cas = torch.softmax(act_inst_cas, dim=2)
        act_cont_cas = torch.softmax(act_cont_cas, dim=2)
        act_back_cas = torch.softmax(act_back_cas, dim=2)
        
        act_cas = torch.softmax(act_cas, dim=2)
        
        _, sorted_act_inst_att_idx = torch.sort(act_inst_att, dim=1, descending=True)
        _, sorted_act_cont_att_idx = torch.sort(act_cont_att, dim=1, descending=True)
        _, sorted_act_back_att_idx = torch.sort(act_back_att, dim=1, descending=True)
        act_inst_feat_idx = sorted_act_inst_att_idx[:, :inst_topk_num, :].expand([-1, -1, self.feature_dim])
        act_cont_feat_idx = sorted_act_cont_att_idx[:, :cont_topk_num, :].expand([-1, -1, self.feature_dim])
        act_back_feat_idx = sorted_act_back_att_idx[:, :back_topk_num, :].expand([-1, -1, self.feature_dim])
        act_inst_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_inst_feat_idx), dim=1)
        act_cont_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_cont_feat_idx), dim=1)
        act_back_feat = torch.mean(torch.gather(embeded_feature_rev, 1, act_back_feat_idx), dim=1)
        
        return act_inst_cls, act_cont_cls, act_back_cls,\
               act_inst_feat, act_cont_feat, act_back_feat,\
               temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas
    
    def forward_strong_sup(self, input_features):
        device = input_features.device
        batch_size, temp_len = input_features.shape[0], input_features.shape[1]

        inst_topk_num = max(temp_len // self.ins_topk_seg, 1)
        cont_topk_num = max(temp_len // self.con_topk_seg, 1)
        back_topk_num = max(temp_len // self.bak_topk_seg, 1)

        input_features = input_features.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_features)
        
        #(act_inst_cls, act_cont_cls, act_back_cls,\
        #act_inst_feat, act_cont_feat, act_back_feat,\
        #temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas)=self.forward_cas_map(self, embeded_feature)

        start = self.x_1d_s(embeded_feature)
        end = self.x_1d_e(embeded_feature)

        #confidence_map = self.x_1d_p(embeded_feature)
        confidence_map = embeded_feature
        confidence_map = self._boundary_matching_layer(confidence_map) #matmul w/ sample mask
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
        return out

    def forward(self, src_input_feature,infer=0):

        if(infer==0):
            # weakly supervised learning
            src_act_inst_cls, src_act_cont_cls, src_act_back_cls,\
            src_act_inst_feat, src_act_cont_feat, src_act_back_feat,\
            src_temp_att, src_act_inst_cas, src_act_cas, _, _ =self.forward_cas_map(src_input_feature)

            # Strong supervision learning
            confidence_map, src_start, src_end = self.forward_strong_sup(src_input_feature)

            # return 
            return  src_act_inst_cls, src_act_cont_cls, src_act_back_cls,\
            src_act_inst_feat, src_act_cont_feat, src_act_back_feat,\
            src_temp_att, src_act_inst_cas, src_act_cas,\
            src_start, src_end, confidence_map
        else:
            act_inst_cls, act_cont_cls, act_back_cls,\
            act_inst_feat, act_cont_feat, act_back_feat,\
            temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas = self.forward_cas_map(src_input_feature)

            return act_inst_cls, act_cont_cls, act_back_cls,\
            act_inst_feat, act_cont_feat, act_back_feat,\
            temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas

class ACMNet_bmn(nn.Module):
    def __init__(self, args):
        super(ACMNet_bmn, self).__init__()
        self.dataset = args.dataset
        self.feature_dim_init = args.feature_dim
        self.feature_dim = args.feature_dim_2
        self.action_cls_num = args.action_cls_num # Only the action categories number.
        self.drop_thresh = args.dropout
        self.ins_topk_seg = args.ins_topk_seg 
        self.con_topk_seg = args.con_topk_seg 
        self.bak_topk_seg = args.bak_topk_seg
        self.r_easy = args.r_easy
        self.r_hard = args.r_hard
        self.m = args.m
        self.M = args.M
        self.tscale = args.sample_segments_num
        self.prop_boundary_ratio = args.prop_boundary_ratio
        self.num_sample = args.num_sample
        self.num_sample_perbin = args.num_sample_perbin

        self.dropout = nn.Dropout(args.dropout)

        self.hidden_dim_1d = self.feature_dim
        #self.hidden_dim_2d = 128
        #self.hidden_dim_3d = 512
        self.hidden_dim_2d = 32
        self.hidden_dim_3d = 128


        self._get_interp1d_mask()

        if self.dataset == "THUMOS":
            self.feature_embedding = nn.Sequential(
                # nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim_init, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            self.feature_embedding = nn.Sequential(
                nn.Dropout(args.dropout),
                nn.Conv1d(in_channels=self.feature_dim_init, out_channels=self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                )
        
        # We introduce three-branch attention, action instance, action context and the irrelevant backgrounds.
        self.att_branch = nn.Conv1d(in_channels=self.feature_dim, out_channels=3, kernel_size=1, padding=0)
        self.snippet_cls = nn.Linear(in_features=self.feature_dim, out_features=(self.action_cls_num + 1))

        # Temporal Evaluation Module
        self.x_1d_s = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.x_1d_e = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, groups=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Proposal Evaluation Module
        #self.x_1d_p = nn.Sequential(
        #    nn.Conv1d(self.hidden_dim_1d, self.hidden_dim_1d, kernel_size=3, padding=1),
        #    nn.ReLU(inplace=True)
        #)
        self.x_3d_p = nn.Sequential(
            nn.Conv3d(self.hidden_dim_1d, self.hidden_dim_3d, kernel_size=(self.num_sample, 1, 1),stride=(self.num_sample, 1, 1)),
            nn.ReLU(inplace=True)
        )
        self.x_2d_p = nn.Sequential(
            nn.Conv2d(self.hidden_dim_3d, self.hidden_dim_2d, kernel_size=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            #nn.Conv2d(self.hidden_dim_2d, self.hidden_dim_2d, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim_2d, 2, kernel_size=1),
            nn.Sigmoid()
        )

        #Domain adaptation
        num_f_maps = 2048
        self.ad_net_base = nn.ModuleList()
        self.ad_net_base += [AdvDomainClsBase(num_f_maps,num_f_maps, "frame",args)]
        self.ad_net_cls = nn.ModuleList()
        self.ad_net_cls += [nn.Linear(num_f_maps, 2)]

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [
            seg_xmin + plen_sample * ii
            for ii in range(num_sample * num_sample_perbin)
        ]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin:(idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask

    def _get_interp1d_mask(self):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for end_index in range(self.tscale):
            mask_mat_vector = []
            for start_index in range(self.tscale):
                if start_index <= end_index:
                    p_xmin = start_index
                    p_xmax = end_index + 1
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_boundary_ratio
                    sample_xmax = p_xmax + center_len * self.prop_boundary_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin, sample_xmax, self.tscale, self.num_sample,
                        self.num_sample_perbin)
                else:
                    p_mask = np.zeros([self.tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3)
        mask_mat = mask_mat.astype(np.float32)
        self.sample_mask = nn.Parameter(torch.Tensor(mask_mat).view(self.tscale, -1), requires_grad=False)
    
    def forward_strong_sup(self, input_features):
        input_features = input_features.permute(0, 2, 1)
        embeded_feature = self.feature_embedding(input_features)
        
        #(act_inst_cls, act_cont_cls, act_back_cls,\
        #act_inst_feat, act_cont_feat, act_back_feat,\
        #temp_att, act_inst_cas, act_cas, act_cont_cas, act_back_cas)=self.forward_cas_map(self, embeded_feature)

        start = self.x_1d_s(embeded_feature).squeeze(1)
        end = self.x_1d_e(embeded_feature).squeeze(1)

        #confidence_map = self.x_1d_p(embeded_feature)
        confidence_map = embeded_feature
        confidence_map = self._boundary_matching_layer(confidence_map) #matmul w/ sample mask
        confidence_map = self.x_3d_p(confidence_map).squeeze(2)
        confidence_map = self.x_2d_p(confidence_map)
        return confidence_map, start, end

    def _boundary_matching_layer(self, x):
        input_size = x.size()
        out = torch.matmul(x, self.sample_mask).reshape(input_size[0],input_size[1],self.num_sample,self.tscale,self.tscale)
        return out

    def forward(self, src_input_feature):

        confidence_map, src_start, src_end = self.forward_strong_sup(src_input_feature)

        return src_start, src_end, confidence_map