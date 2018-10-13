import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch.autograd import Variable

class TripletLoss(nn.Module):
	def __init__(self):
		super(TripletLoss, self).__init__()
		return

	def forward(self, que_emb, pos_emb, neg_emb):
		pdist = nn.PairwiseDistance(p = 2)
		matched = pdist(que_emb, pos_emb)
		mismatched = pdist(que_emb, neg_emb)
		g = torch.ones(1)
		loss = torch.max(torch.zeros(1), g + matched - mismatched)
		loss = torch.mean(loss)
		return loss