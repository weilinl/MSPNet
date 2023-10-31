import torch

def LD_loss(feature1, feature2):
  distance = torch.abs(feature1 - feature2)
  distance[distance < 0.01] = 100
  loss_contrastive = torch.mean(torch.exp(-distance))
  return loss_contrastive


def tramsform(mr):
  b, c = mr.size()
  mr = mr.reshape(int(b / 2), 2, c)
  (mra, mrb) = mr.chunk(2, dim=0)
  mr = torch.cat((mrb, mra), dim=0)
  mr = mr.reshape(b, c)
  return mr
