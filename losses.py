import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class TRLoss(nn.Module):
    def __init__(self, gamma, eps=1e-8, num_labels=-1):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        self.num_labels = num_labels

    def forward(self, logits, labels):
        n, num_class = labels.size()
        th_logits = logits[:, 0].unsqueeze(-1).expand(-1, num_class - 1).unsqueeze(-1)
        re_logits = logits[:, 1:].unsqueeze(-1)
        rank_logits = torch.cat([th_logits, re_logits], dim=-1)
        rank_prob = torch.softmax(rank_logits, dim=-1)
        th_prob = rank_prob[:, :, 0].clamp(min=self.eps)
        re_prob = rank_prob[:, :, 1].clamp(min=self.eps)
        # re_prob = (re_prob + self.shift).clamp(min=self.eps, max=1)
        p_mask = labels[:, 1:]
        n_mask = 1 - p_mask
        pos_loss = p_mask * (1 - re_prob).pow(self.gamma) * torch.log(re_prob)
        neg_loss = n_mask * torch.log(th_prob)
        return -(pos_loss + neg_loss).sum() / n

    def get_label(self, logits):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if self.num_labels > 0:
            top_v, _ = torch.topk(logits, self.num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output


class AFLoss(nn.Module):
    def __init__(self, gamma_pos, gamma_neg, num_labels=-1):
        super().__init__()
        threshod = nn.Threshold(0, 0)
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.num_labels = num_labels

    def forward(self, logits, labels):
        # 
        # TH label 
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0
        label_idx = labels.sum(dim=1)

        two_idx = torch.where(label_idx == 2)[0]
        pos_idx = torch.where(label_idx > 0)[0]

        neg_idx = torch.where(label_idx == 0)[0]

        p_mask = labels + th_label
        n_mask = 1 - labels
        neg_target = 1 - p_mask

        num_ex, num_class = labels.size()
        num_ent = int(np.sqrt(num_ex))
        # Rank each positive class to TH
        logit1 = logits - neg_target * 1e30
        logit0 = logits - (1 - labels) * 1e30

        # Rank each class to threshold class TH
        th_mask = torch.cat(num_class * [logits[:, :1]], dim=1)
        logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1)
        log_probs = F.log_softmax(logit_th, dim=1)
        probs = torch.exp(F.log_softmax(logit_th, dim=1))

        # Probability of relation class to be positive (1)
        prob_1 = probs[:, 0, :]
        # Probability of relation class to be negative (0)
        prob_0 = probs[:, 1, :]
        prob_1_gamma = torch.pow(prob_1, self.gamma_neg)
        prob_0_gamma = torch.pow(prob_0, self.gamma_pos)
        log_prob_1 = log_probs[:, 0, :]
        log_prob_0 = log_probs[:, 1, :]

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        rank2 = F.log_softmax(logit2, dim=-1)

        loss1 = - (log_prob_1 * (1 + prob_0_gamma) * labels)  # (1650, 97)

        loss2 = -(rank2 * th_label).sum(1)  # (1650,)

        loss = 1.0 * loss1.sum(1).mean() + 1.0 * loss2.mean()

        return loss

    def get_label(self, logits):
        th_logit = logits[:, 0].unsqueeze(1) * 1.0
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if self.num_labels > 0:
            top_v, _ = torch.topk(logits, self.num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output


class NCRLoss(nn.Module):
    def __init__(self, shift=0.0, isReg=True, eps=1e-8, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.isReg = isReg
        self.eps = eps
        self.reduction = reduction

    def compute_CE(self, x, y):
        """
        
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Margin Shifting
        if self.shift is not None and self.shift > 0:
            xs_neg = (xs_neg + self.shift).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        return -loss.sum()

    def forward(self, logits, labels):
        # Logit margin for pre-defined relations
        rel_margin = logits[:, 1:] - logits[:, 0].unsqueeze(1)
        loss = self.compute_CE(rel_margin.float(), labels[:, 1:].float())

        if self.isReg:  # Enable margin regularization
            # Logit margin for the none class label
            na_margin = logits[:, 0] - logits[:, 1:].mean(-1)
            loss += self.compute_CE(na_margin.float(), labels[:, 0].float())

        if self.reduction == "mean":
            loss /= labels.shape[0]

        return loss

    def get_label(self, logits, num_labels=-1):
        
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output
