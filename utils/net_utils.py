from turtle import forward
import torch 
import random 
import numpy as np 
import torch.nn as nn 

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    
def weights_init(model):
    if isinstance(model, nn.Conv2d):
        model.weights.data.normal_(0.0, 0.001)
    elif isinstance(model, nn.Linear):
        model.weights.data.normal_(0.0, 0.001)

    
class ACMLoss(nn.Module):
    
    def __init__(self, lamb1=2e-3, lamb2=5e-5, lamb3=2e-4, dataset="THUMOS14"):
        super(ACMLoss, self).__init__()
        
        self.dataset = dataset
        self.lamb1 = lamb1 # att_norm_loss param 
        self.lamb2 = lamb2
        self.lamb3 = lamb3 
        self.feat_margin = 50  #50
        
    def cls_criterion(self, inputs, label):
        return - torch.mean(torch.sum(torch.log(inputs) * label, dim=1))
    
    def forward(self, act_inst_cls, act_cont_cls, act_back_cls, vid_label, temp_att=None,\
                act_inst_feat=None, act_cont_feat=None, act_back_feat=None, temp_cas=None):
        
        device = act_inst_cls.device 
        batch_size = act_inst_cls.shape[0]
        
        act_inst_label = torch.hstack((vid_label, torch.zeros((batch_size, 1), device=device)))
        act_cont_label = torch.hstack((vid_label, torch.ones((batch_size, 1), device=device)))
        act_back_label = torch.hstack((torch.zeros_like(vid_label), torch.ones((batch_size, 1), device=device)))
        
        act_inst_label = act_inst_label / torch.sum(act_inst_label, dim=1, keepdim=True)
        act_cont_label = act_cont_label / torch.sum(act_cont_label, dim=1, keepdim=True)
        act_back_label = act_back_label / torch.sum(act_back_label, dim=1, keepdim=True)
        
        act_inst_loss = self.cls_criterion(act_inst_cls, act_inst_label)
        act_cont_loss = self.cls_criterion(act_cont_cls, act_cont_label)
        act_back_loss = self.cls_criterion(act_back_cls, act_back_label)
        
        # Guide Loss
        guide_loss = torch.sum(torch.abs(1 - temp_cas[:, :, -1] - temp_att[:, :, 0].detach()), dim=1).mean()

        # Feat Loss
        act_inst_feat_norm = torch.norm(act_inst_feat, p=2, dim=1)
        act_cont_feat_norm = torch.norm(act_cont_feat, p=2, dim=1)
        act_back_feat_norm = torch.norm(act_back_feat, p=2, dim=1)
        
        feat_loss_1 = self.feat_margin - act_inst_feat_norm + act_cont_feat_norm
        feat_loss_1[feat_loss_1 < 0] = 0
        feat_loss_2 = self.feat_margin - act_cont_feat_norm + act_back_feat_norm
        feat_loss_2[feat_loss_2 < 0] = 0
        feat_loss_3 = act_back_feat_norm
        feat_loss = torch.mean((feat_loss_1 + feat_loss_2 + feat_loss_3)**2)

        # Sparse Att Loss
        # att_loss = torch.sum(temp_att[:, :, 0], dim=1).mean() + torch.sum(temp_att[:, :, 1], dim=1).mean() 
        sparse_loss = torch.sum(temp_att[:, :, :2], dim=1).mean()
        
        if self.dataset == "THUMOS14":
            cls_loss = 1.0 * act_inst_loss + 1.0 * act_cont_loss + 1.0 * act_back_loss
        else:
            # We add more strong regularization operation to the ActivityNet dataset since this dataset contains much more diverse videos.
            cls_loss = 5.0 * act_inst_loss + 1.0 * act_cont_loss + 1.0 * act_back_loss
            
        add_loss = self.lamb1 * guide_loss + self.lamb2 * feat_loss + self.lamb3 * sparse_loss
        
        loss = cls_loss + add_loss
        
        loss_dict = {}
        loss_dict["act_inst_loss"] = act_inst_loss.cpu().item()
        loss_dict["act_cont_loss"] = act_cont_loss.cpu().item()
        loss_dict["act_back_loss"] = act_back_loss.cpu().item()
        loss_dict["guide_loss"] = guide_loss.cpu().item()
        loss_dict["feat_loss"] = feat_loss.cpu().item()
        loss_dict["sparse_loss"] = sparse_loss.cpu().item()
        
        return loss, loss_dict

class DomAdpLoss(nn.Module):
    def __init__(self):
        self.ce_d = nn.CrossEntropyLoss(reduction='none')

    def forward(self):
        # ------ Adversarial loss ------ #
        num_class_domain = 2
        loss_adv = 0
        for c in range(num_class_domain):
            pred_d_class = pred_d_stage[:, c, :]  # (batch x frame#, 2)
            label_d_class = label_d_stage[:, c]  # (batch x frame#)

            loss_adv_class = self.ce_d(pred_d_class, label_d_class)
            if weighted_domain_loss == 'Y' and multi_adv[1] == 'Y':  # weighted by class prediction
                if ps_lb == 'soft':
                    loss_adv_class *= classweight_stage[:, c].detach()
                elif ps_lb == 'hard':
                    loss_adv_class *= classweight_stage_hardmask[:, c].detach()

            loss_adv += loss_adv_class.mean()

        loss += loss_adv

        #if 'rev_grad' in DA_adv_video:
        loss_adv_video = self.ce_d(pred_d_video_stage, label_d_video_stage)
        loss += loss_adv_video.mean()
        
        return

class BMNLoss(nn.Module):
    def __init__(self):
        super(BMNLoss, self).__init__()
        
        self.ce_d = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self,pred_start, pred_end, gt_start, gt_end):
        #pred_bm = confidence_map, gt_iou_map = label_confidence
        tem_loss = self.tem_loss_func(pred_start, pred_end, gt_start, gt_end)

        loss = tem_loss 
        return loss, tem_loss


    def tem_loss_func(self,pred_start, pred_end, gt_start, gt_end):
        #weighted binary logistic regression loss function, following BSN
        def bi_loss(pred_score, gt_label):
            device = pred_score.device
            pred_score = pred_score.view(-1)
            gt_label = gt_label.view(-1)
            pmask = (gt_label > 0.5).float() #bi => value is {0,1} => 0 or 1
            num_entries = len(pmask)
            num_positive = torch.sum(pmask)
            ratio = num_entries / num_positive #alpha_+
            #ratio / (ratio - 1) = alpha_-
            coef_0 = 0.5 * ratio / (ratio - 1)
            coef_1 = 0.5 * ratio #alpha_+
            epsilon = 0.000001

            pmask = pmask.to(device)

            loss_pos = coef_1 * torch.log(pred_score + epsilon) * pmask
            loss_neg = coef_0 * torch.log(1.0 - pred_score + epsilon) * (1.0 - pmask)
            loss = -1 * torch.mean(loss_pos + loss_neg)
            return loss

        loss_start = bi_loss(pred_start, gt_start)
        loss_end = bi_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss



class SniCoLoss(nn.Module):
    #HA refinement aims to transform the hard action snippet features by driving hard action and easy action snippets compactly in feature space"
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07): #Negative cross entropy
        # q size = nx2d = ndata x channel
        q = nn.functional.normalize(q, dim=1) # normalized unit sphere to prevent collapsing or expanding
        k = nn.functional.normalize(k, dim=1) # normalized unit sphere to prevent collapsing or expanding
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1) # normalized unit sphere to prevent collapsing or expanding
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) #positive query exp(xT.x+) nc.cn
        l_neg = torch.einsum('nc,nck->nk', [q, neg]) # sum(negative query xT.xs-)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T #Temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1),  #x
            torch.mean(contrast_pairs['EA'], 1),  #x+
            contrast_pairs['EB'] #x-
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1), #x
            torch.mean(contrast_pairs['EB'], 1), #x+
            contrast_pairs['EA'] #x-
        )

        loss = HA_refinement + HB_refinement
        return loss