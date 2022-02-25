import os 
import json 
import torch 
import argparse 
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset 
import torch.utils.data as data
from torch.functional import F

import numpy as np


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard

def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores

def uniform_sample(input_feature, sample_len):
        
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)

    if input_len <= sample_len and input_len > 1: #shorter than T
        sample_idxs = np.arange(input_len)# idxs follow vid
    else: #larger than T
        if input_len == 1:
            sample_len = 2
        sample_scale = input_len / sample_len
        sample_idxs = np.arange(sample_len) * sample_scale
        sample_idxs = np.floor(sample_idxs)

    return input_feature[sample_idxs.astype(np.int), :]
    
def random_sample(input_feature, sample_len):
    
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)
    
    if input_len < sample_len:
        sample_idxs = np.random.choice(input_len, sample_len, replace=True)
        sample_idxs = np.sort(sample_idxs)
    elif input_len > sample_len:
        sample_idxs = np.arange(sample_len) * input_len / sample_len
        for i in range(sample_len-1):
            sample_idxs[i] = np.random.choice(range(np.int(sample_idxs[i]), np.int(sample_idxs[i+1] + 1)))
        sample_idxs[-1] = np.random.choice(np.arange(sample_idxs[-2], input_len))
    else:
        sample_idxs = np.arange(input_len)
    
    return input_feature[sample_idxs.astype(np.int), :]

def consecutive_sample(input_feature, sample_len):
    
    input_len = input_feature.shape[0]
    assert sample_len > 0, "WRONG SAMPLE_LEN {}, THIS PARAM MUST BE GREATER THAN 0.".format(sample_len)
    
    if input_len >= sample_len:
        sample_idx = np.random.choice((input_len - sample_len))
        return input_feature[sample_idx:(sample_idx+sample_len), :]
    
    elif input_len < sample_len:
        empty_features = np.zeros((sample_len - input_len, input_feature.shape[1]))
        return np.concatenate((input_feature, empty_features), axis=0)

class ACMDataset(Dataset):
    
    def __init__(self, args, phase="train", sample="random"):
        
        self.phase = phase 
        self.sample = sample
        self.data_dir = args.data_dir 
        self.sample_segments_num = args.sample_segments_num
        
        with open(os.path.join(self.data_dir, "gt.json")) as gt_f:
            self.gt_dict = json.load(gt_f)["database"]
            
        if self.phase == "train":
            self.feature_dir = os.path.join(self.data_dir, "train")
            self.data_list = list(open(os.path.join(self.data_dir, "split_train.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        else:
            self.feature_dir = os.path.join(self.data_dir, "test")
            self.data_list = list(open(os.path.join(self.data_dir, "split_test.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        
        self.class_name_lst = args.class_name_lst
        self.action_class_idx_dict = {action_cls:idx for idx, action_cls in enumerate(self.class_name_lst)}
        
        self.action_class_num = args.action_cls_num
        
        self.get_label()
        
    def get_label(self):
        
        self.label_dict = {}
        for item_name in self.data_list:
            
            item_anns_list = self.gt_dict[item_name]["annotations"]
            item_label = np.zeros(self.action_class_num)
            for ann in item_anns_list:
                ann_label = ann["label"]
                item_label[self.action_class_idx_dict[ann_label]] = 1.0
            
            self.label_dict[item_name] = item_label

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        
        vid_name = self.data_list[idx]
        vid_label = self.label_dict[vid_name]
        vid_duration = self.gt_dict[vid_name]["duration"]
        con_vid_feature = np.load(os.path.join(self.feature_dir, vid_name+".npy"))
        
        vid_len = con_vid_feature.shape[0]
        
        # in sampling part, temporal length of input snippet are adjusted to T
        # sample_segments_num = 75 if ActNet
        if self.sample == "random":
            con_vid_spd_feature = random_sample(con_vid_feature, self.sample_segments_num)
        else:
            con_vid_spd_feature = uniform_sample(con_vid_feature, self.sample_segments_num)
        
        con_vid_spd_feature = torch.as_tensor(con_vid_spd_feature.astype(np.float32)) 
        
        vid_label_t = torch.as_tensor(vid_label.astype(np.float32))
        
        if self.phase == "train":
            return con_vid_spd_feature, vid_label_t 
        else:
            return vid_name, con_vid_spd_feature, vid_label_t, vid_len, vid_duration


def build_dataset(args, phase="train", sample="random"):
    
    return ACMDataset(args, phase, sample)

class SourceVidDataset(data.Dataset):
    
    def __init__(self, args, phase="train", sample="random", get_bound=False):
        
        self.sample_segments_num = args.sample_segments_num
        self.temporal_scale = self.sample_segments_num
        self.temporal_gap = 1. / self.temporal_scale

        self.phase = phase
        self.sample = sample
        self.data_dir = args.src_data_dir 
        self.sample_segments_num = args.sample_segments_num
        self.get_bound = get_bound

        with open(os.path.join(self.data_dir, "gt.json")) as gt_f:
            self.gt_dict = json.load(gt_f)["database"]
            
        if self.phase == "train":
            self.feature_dir = os.path.join(self.data_dir, "train")
            self.data_list = list(open(os.path.join(self.data_dir, "split_train.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        else:
            self.feature_dir = os.path.join(self.data_dir, "test")
            self.data_list = list(open(os.path.join(self.data_dir, "split_test.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        
        self.class_name_lst = args.class_name_lst
        self.action_class_idx_dict = {action_cls:idx for idx, action_cls in enumerate(self.class_name_lst)}
        
        self.action_class_num = args.action_cls_num

        # Additional
        
        self.anchor_xmin = np.array([self.temporal_gap * (i - 0.5) for i in range(self.temporal_scale)])
        self.anchor_xmax = np.array([self.temporal_gap * (i + 0.5) for i in range(self.temporal_scale)])
        self.get_label()

    def __getitem__(self, idx):
        
        vid_name = self.data_list[idx]
        vid_label = self.label_dict[vid_name]
        vid_duration = self.gt_dict[vid_name]["duration"] #action duration in seconds
        con_vid_feature = np.load(os.path.join(self.feature_dir, vid_name+".npy"))
        
        vid_len = con_vid_feature.shape[0] #number of frames
        
        if self.sample == "random":
            con_vid_spd_feature = random_sample(con_vid_feature, self.sample_segments_num)
            con_vid_spd_feature = torch.as_tensor(con_vid_spd_feature.astype(np.float32)) #input_feature
        elif self.sample == "interpolate":
            con_vid_spd_feature = torch.Tensor(con_vid_feature)
            con_vid_spd_feature = torch.transpose(con_vid_spd_feature, 0, 1)
            if con_vid_spd_feature.shape[0]!=self.temporal_scale: # rescale to fixed shape
                con_vid_spd_feature = F.interpolate(con_vid_spd_feature.unsqueeze(0), size=self.temporal_scale, mode='linear',align_corners=False)[0,...]
            con_vid_spd_feature = torch.transpose(con_vid_spd_feature, 0, 1)
        else:
            con_vid_spd_feature = uniform_sample(con_vid_feature, self.sample_segments_num)
            con_vid_spd_feature = torch.as_tensor(con_vid_spd_feature.astype(np.float32)) #input_feature
        
        vid_label_t = torch.as_tensor(vid_label.astype(np.float32)) #video-level label
        
        label_start, label_end, sample_bkg_idx, sample_act_idx, gt_iou_map = self._get_train_prop_label(idx,self.anchor_xmin,self.anchor_xmax)
        
        init_tensor = torch.zeros(self.sample_segments_num,2048)
        sample_bkg_idx = np.clip(sample_bkg_idx, 0, len(con_vid_spd_feature)-1)
        if sample_bkg_idx.size>1:
            sample_bkg_feat = con_vid_spd_feature[sample_bkg_idx,:]
            init_tensor[:len(sample_bkg_idx),:]=sample_bkg_feat

        sample_bkg_feat = init_tensor


        init_tensor = torch.zeros(self.sample_segments_num,2048)
        sample_act_idx = np.clip(sample_act_idx, 0, len(con_vid_spd_feature)-1)
        if sample_act_idx.size>1:
            sample_act_feat = con_vid_spd_feature[sample_act_idx,:]
            init_tensor[:len(sample_act_idx),:]=sample_act_feat
            
        sample_act_feat = init_tensor

        if self.phase == "train":
            return con_vid_spd_feature, vid_label_t, gt_iou_map, label_start, label_end, sample_bkg_feat, sample_act_feat
        else:
            if self.get_bound == True:
                return vid_name, con_vid_spd_feature, vid_label_t,gt_iou_map, label_start, label_end, vid_len, vid_duration
            else:
                return vid_name, con_vid_spd_feature, vid_label_t, vid_len, vid_duration

        
    def get_label(self):
        
        self.label_dict = {}
        for item_name in self.data_list:
            
            item_anns_list = self.gt_dict[item_name]["annotations"]
            item_label = np.zeros(self.action_class_num)
            for ann in item_anns_list:
                ann_label = ann["label"]
                item_label[self.action_class_idx_dict[ann_label]] = 1.0
            
            self.label_dict[item_name] = item_label
            #additional
            #video_second = self.gt_dict[item_name]["duration"]

    def __len__(self):
        return len(self.data_list)
    
    def _get_train_prop_label(self,index, anchor_xmin, anchor_xmax):
        video_name = self.data_list[index]
        video_info = self.gt_dict[video_name]
        video_labels = video_info['annotations']
        video_second = float(video_info['duration'])
        corrected_second = video_second
        temporal_scale = self.sample_segments_num
        temporal_gap = 1. / temporal_scale

        # change the measurement from second to percentage
        # gt_bbox = active window in percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            tmp_info = video_labels[j]
            tmp_start = max(min(1, tmp_info['segment'][0] / corrected_second), 0)
            tmp_end = max(min(1, tmp_info['segment'][1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])

        #####################################################################################################
        # generate R_s and R_e
        # starting and ending region
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        gt_iou_map = np.zeros([self.temporal_scale, self.temporal_scale])
        for i in range(self.temporal_scale):
            for j in range(i, self.temporal_scale):
                gt_iou_map[i, j] = np.max(
                    iou_with_anchors(i * self.temporal_gap, (j + 1) * self.temporal_gap, gt_xmins, gt_xmaxs))
        gt_iou_map = torch.Tensor(gt_iou_map)


        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)

        ### ==== Additional ====
        gt_start_nonact = gt_bbox[:,1]+0.5*temporal_gap # get end border of end region
        gt_end_nonact = gt_bbox[:,0]-0.5*temporal_gap # get start border of start region

        ls_start = gt_start_nonact[1:]
        ls_end = gt_end_nonact[:-1]

        gt_nonact_bbox = [(0,gt_end_nonact[0])]
        gt_nonact_bbox.extend(list(zip(ls_start,ls_end)))
        gt_nonact_bbox.append((gt_start_nonact[-1],1))


        #anchor_xs = list(zip(anchor_xmin,anchor_xmax))
        #Assign backgroun label to each segment
        list_bkg_lbl=[]
        for id_anc in range(len(anchor_xmax)):
        #for anchor_now in anchor_xs:
            tem_bak = 0
            for bound in gt_nonact_bbox:
                if((max(0,anchor_xmin[id_anc])>=bound[0]) and (min(anchor_xmax[id_anc],1)<=bound[1])):
                    tem_bak = 1
                    break
            list_bkg_lbl.append(tem_bak)

        list_bkg_lbl = torch.Tensor(list_bkg_lbl)
        src_bkg_list = list_bkg_lbl.nonzero()
        src_bkg_idx=src_bkg_list.numpy().squeeze()

        src_act_list = (list_bkg_lbl == 0).nonzero()
        src_act_idx=src_act_list.numpy().squeeze()
        return match_score_start, match_score_end, src_bkg_idx, src_act_idx, gt_iou_map

def build_src_dataset(args, phase="train", sample="random", get_bound=False):
    
    return SourceVidDataset(args, phase, sample, get_bound)

class TargetVidDataset(Dataset):
    
    def __init__(self, args, phase="train", sample="random"):
        
        self.phase = phase 
        self.sample = sample
        self.data_dir = args.tgt_data_dir 
        self.sample_segments_num = args.sample_segments_num
        
        with open(os.path.join(self.data_dir, "gt.json")) as gt_f:
            self.gt_dict = json.load(gt_f)["database"]
            
        if self.phase == "train":
            self.feature_dir = os.path.join(self.data_dir, "train")
            self.data_list = list(open(os.path.join(self.data_dir, "split_train.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        else:
            self.feature_dir = os.path.join(self.data_dir, "test")
            self.data_list = list(open(os.path.join(self.data_dir, "split_test.txt")))
            self.data_list = [item.strip() for item in self.data_list]
        
        self.class_name_lst = args.class_name_lst
        self.action_class_idx_dict = {action_cls:idx for idx, action_cls in enumerate(self.class_name_lst)}
        
        self.action_class_num = args.action_cls_num
        
        self.get_label()
        
    def get_label(self):
        
        self.label_dict = {}
        for item_name in self.data_list:
            
            item_anns_list = self.gt_dict[item_name]["annotations"]
            item_label = np.zeros(self.action_class_num)
            for ann in item_anns_list:
                ann_label = ann["label"]
                item_label[self.action_class_idx_dict[ann_label]] = 1.0
            
            self.label_dict[item_name] = item_label

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        
        vid_name = self.data_list[idx]
        vid_label = self.label_dict[vid_name]
        vid_duration = self.gt_dict[vid_name]["duration"]
        con_vid_feature = np.load(os.path.join(self.feature_dir, vid_name+".npy"))
        
        vid_len = con_vid_feature.shape[0]
        
        # in sampling part, temporal length of input snippet are adjusted to T
        # sample_segments_num = 75 if ActNet
        if self.sample == "random":
            con_vid_spd_feature = random_sample(con_vid_feature, self.sample_segments_num)
        else:
            con_vid_spd_feature = uniform_sample(con_vid_feature, self.sample_segments_num)
        
        con_vid_spd_feature = torch.as_tensor(con_vid_spd_feature.astype(np.float32)) 
        
        vid_label_t = torch.as_tensor(vid_label.astype(np.float32))
        
        if self.phase == "train":
            return con_vid_spd_feature, vid_label_t 
        else:
            return vid_name, con_vid_spd_feature, vid_label_t, vid_len, vid_duration

def build_tgt_dataset(args, phase="train", sample="random", get_bound=False):
    
    return TargetVidDataset(args, phase, sample)



def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data

def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data