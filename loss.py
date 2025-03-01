import torch
from torch import nn

class stage1_loss(nn.Module):
    def __init__(self, args):
        super(stage1_loss, self).__init__()
        self.args = args

    def forward(self, promote_drug_1, promote_drug_2, labels):
        sim = torch.cosine_similarity(promote_drug_1, promote_drug_2, dim=1)
        sim_loss = (sim - labels)**2

        return sim_loss.mean()