import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch.autograd import Variable

class TripletLoss(nn.Module):
	def __init__(self):
		return

	def forward(self, que_emb, pos_emb, neg_emb):
		"""
		Args:
			que_emb: the image embedding of the query image
			pos_emb: the image embedding of positive image
			neg_emb: the image embedding of negative image
		"""
		pdist = nn.PairwiseDistance(p = 2)
		matched = pdist(que_emb, pos_emb)
		mismatched = pdist(que_emb, neg_emb)
		g = torch.ones(1)
		loss = torch.max(torch.zeros(1), g + matched - mismatched)
		return loss

# # Test
# test = TripletLoss()
# q = Variable(torch.randn(1, 3), requires_grad = True)
# p = Variable(torch.randn(1, 3), requires_grad = True)
# n = Variable(torch.randn(1, 3), requires_grad = True)
# loss = test.forward(q, p, n)
# print(loss)
