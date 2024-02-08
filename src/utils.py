import torch
from torch import nn, optim
import numpy as np
from typing import Any, Callable, List, Tuple, Union
from torch.optim import Optimizer
from transformers import AdamW
import torch.nn.functional as F
from functools import partial
import networkx as nx
from scipy.stats import rankdata

class ContrastiveLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.score_func     = kwargs['score_function']
        self.tau            = kwargs['temperature_scalar']
        if self.score_func == 'dualentropy':
            self.criterion = nn.CrossEntropyLoss().to(kwargs['rank'])
        elif self.score_func == 'cosinesimilarity':
            self.criterion  = nn.CosineSimilarity(dim=2).to(kwargs['rank'])
            self.logsoftmax = nn.LogSoftmax(1).to(kwargs['rank'])
        elif self.score_func == 'euclidean':
            self.logsoftmax = nn.LogSoftmax(1).to(kwargs['rank'])
        elif self.score_func == 'none':
            pass
        else:
            raise ValueError("Invalid Score Function for Contrastive Learning")

    def forward(self, view1: Union[torch.Tensor, None], 
                      view2: Union[torch.Tensor, None], 
                      stoic: Union[torch.Tensor, None]):
        if not isinstance(view1, torch.Tensor):
            return torch.cuda.FloatTensor([0.0])

        if not isinstance(view2, torch.Tensor):
            return torch.cuda.FloatTensor([0.0])

        if self.score_func == 'dualentropy':
            X = F.normalize(view1,dim=-1)
            Y = F.normalize(view2,dim=-1)
            B = X.size()[0]

            logits          = torch.mm(X, Y.transpose(1, 0))
            logits          = torch.div(logits, self.tau)
            labels          = torch.arange(B).long().to(logits.device)  
            loss1           = self.criterion(logits, labels)

            logits          = torch.mm(Y, X.transpose(1, 0))
            logits          = torch.div(logits, self.tau)
            labels          = torch.arange(B).long().to(logits.device)  
            loss2           = self.criterion(logits, labels)

            loss            = 0.5*loss1 + 0.5*loss2

        elif self.score_func == 'cosinesimilarity':
            anchor          = view1
            positive        = view2
            negatives       = view2
            exp_size        = negatives.size(0)
            negatives       = negatives.unsqueeze(0).repeat(exp_size,1,1)
            anchor_expanded = anchor.unsqueeze(1).repeat(1, exp_size+1, 1)
            target_expanded = torch.cat([positive.unsqueeze(1), negatives], 1)
            scores          = self.criterion(anchor_expanded, target_expanded)
            scores          = -1.0 * self.logsoftmax(scores/self.tau)

            loss            = scores[:, 0].mean()

        elif self.score_func == 'euclidean':
            anchor          = view1
            positive        = view2
            negatives       = view2
            exp_size        = negatives.size(0)
            negatives       = negatives.unsqueeze(0).repeat(exp_size,1,1)
            anchor_expanded = anchor.unsqueeze(1).repeat(1, exp_size+1, 1)
            target_expanded = torch.cat([positive.unsqueeze(1), negatives], 1)
            scores          = -1.0 * (anchor_expanded - target_expanded).abs().pow(2).sum(2).sqrt() 
            scores          = -1.0 * self.logsoftmax(scores/self.tau)

            loss            = scores[:, 0].mean()

        elif self.score_func == 'none':
            return torch.cuda.FloatTensor([0.0])

        else:
            raise

        return loss

class RegressionLoss(nn.Module):
    def __init__(self, **kwargs):
        super(RegressionLoss, self).__init__()
        self.task_name = kwargs['task']
        self.criterion = nn.MSELoss().to(kwargs['rank'])

    def forward(self, batch):

        return self.criterion(batch[f'{self.task_name}/pred'],batch[f'{self.task_name}/true'])

class ClassificationLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ClassificationLoss, self).__init__()
        self.task_name = kwargs['task']
        self.rank      = kwargs['rank']
        self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(kwargs['rank'])

    def forward(self, batch):
        yhat    = batch[f'{self.task_name}/pred'].view(-1)
        y       = batch[f'{self.task_name}/true'].view(-1)
        isvalid = y ** 2 > 0
        loss    = self.criterion(yhat, (y+1)/2)
        loss    = torch.where(isvalid,loss,torch.zeros(loss.shape).to(self.rank).to(loss.dtype))

        return loss.sum() / isvalid.sum()


class MultiClassificationLoss(nn.Module):
    def __init__(self, **kwargs):
        super(MultiClassificationLoss, self).__init__()
        self.task_name = kwargs['task']
        self.rank      = kwargs['rank']
        self.criterion = nn.BCEWithLogitsLoss(reduction='none').to(kwargs['rank'])

    def forward(self, batch):
        yhat    = batch[f'{self.task_name}/pred'].view(-1)
        y       = batch[f'{self.task_name}/true'].view(-1)
        isvalid = y ** 2 > 0
        loss    = self.criterion(yhat, (y+1)/2)
        loss    = torch.where(isvalid,loss,torch.zeros(loss.shape).to(self.rank).to(loss.dtype))

        return loss.sum() / isvalid.sum()

class DummyScheduler:
    def __init__(self):
        x = 0
    def step(self):
        return 

def numpify(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list):
        return [element.detach().cpu().numpy() for element in tensor]
    else:
        return tensor