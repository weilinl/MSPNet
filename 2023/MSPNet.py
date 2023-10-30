#!/usr/bin/env python3
from sklearn import svm
import os
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

#30 SRM filtes
from srm_filter_kernel import all_normalized_hpf_list
#Global covariance pooling
#from MPNCOV.python import MPNCOV
import MPNCOV

IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2

EPOCHS = 200
LR = 0.01

WEIGHT_DECAY = 5e-4

TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [80, 140, 180]

OUTPUT_PATH = Path(__file__).stem

def pairedloss(feature1, feature2):
  distance = torch.abs(feature1 - feature2)
  distance[distance < 0.01] = 100
  loss_contrastive = torch.mean(torch.exp(-distance))
  return loss_contrastive


def cs_sc(mr):
  b, c = mr.size()
  mr = mr.reshape(int(b / 2), 2, c)
  mr = torch.transpose(mr, 1, 0)
  (mra, mrb) = mr.chunk(2, dim=0)
  mr = torch.cat((mrb, mra), dim=0)
  mr = torch.transpose(mr, 1, 0)
  mr = mr.reshape(b, c)
  return mr


#Truncation operation
class TLU(nn.Module):
  def __init__(self, threshold):
    super(TLU, self).__init__()

    self.threshold = threshold

  def forward(self, input):
    output = torch.clamp(input, min=-self.threshold, max=self.threshold)

    return output


class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    #Load 30 SRM Filters
    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)

    hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)
    self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight

    #Truncation, threshold = 3
    self.tlu = TLU(3.0)

  def forward(self, input):

    output = self.hpf(input)
    output = self.tlu(output)

    return output


class CBR(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(CBR, self).__init__()

    self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=False)
    self.bn = nn.BatchNorm2d(out_dim)
    self.act = nn.ReLU()

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.act(x)

    return x


class SFEBlock(nn.Module):
  def __init__(self, in_channels):
    super(SFEBlock, self).__init__()

    self.conv1 = CBR(in_channels, 2 * in_channels)
    self.conv1s = CBR(2 * in_channels, 2 * in_channels)
    self.conv2 = CBR(2 * in_channels, 2 * in_channels)
    self.conv2s = CBR(2 * in_channels, 2 * in_channels)
    self.conv3 = CBR(2 * in_channels,  4 * in_channels)
    self.conv3s = CBR(4 * in_channels, 4 * in_channels)
    self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)


  def forward(self, out):
    out = self.avg(out)
    out = self.conv1(out)
    out = self.conv1s(out)
    out = self.avg(out)
    out = self.conv2(out)
    out = self.conv2s(out)
    out = self.avg(out)
    out = self.conv3(out)
    out = self.conv3s(out)
    return out


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.group1 = HPF()

    self.group20 = nn.Sequential(
      nn.Conv2d(30, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
    )
    self.group21 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
    )
    self.group22 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
    )
    self.group23 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
    )
    self.group2_cf = SFEBlock(32)

    self.group3 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
    )
    self.group3_cf = SFEBlock(32)

    self.group4 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
    )
    self.group4_cf = SFEBlock(32)


    self.fc2 = nn.Linear(int(128 * (128 + 1) / 2), 256)
    self.bn2 = nn.BatchNorm1d(256)
    self.act2 = nn.ReLU()
    self.fc21 = nn.Linear(256, 2)

    self.fc3 = nn.Linear(int(128 * (128 + 1) / 2), 256)
    self.bn3 = nn.BatchNorm1d(256)
    self.act3 = nn.ReLU()
    self.fc31 = nn.Linear(256, 2)

    self.fc4 = nn.Linear(int(128 * (128 + 1) / 2), 256)
    self.bn4 = nn.BatchNorm1d(256)
    self.act4 = nn.ReLU()
    self.fc41 = nn.Linear(256, 2)


  def forward(self, input):
    x = input

    x = self.group1(x)
    x20 = self.group20(x)
    x21 = self.group21(x20)
    x22 = self.group22(x21)
    x23 = self.group23(x22)
    x2 = self.group2_cf(x23)

    x3 = x22 + x23
    x30 = self.group3(x3)
    x3 = self.group3_cf(x30)

    x4 = x21 + x30
    x4 = self.group4(x4)
    x4 = self.group4_cf(x4)

    x2 = MPNCOV.CovpoolLayer(x2)
    x2 = MPNCOV.SqrtmLayer(x2, 5)
    x2 = MPNCOV.TriuvecLayer(x2)
    x2 = x2.view(x2.size(0), -1)
    x2 = self.fc2(x2)
    x2 = self.bn2(x2)
    x2t = self.act2(x2)
    x2 = self.fc21(x2t)

    #Global covariance pooling
    x3 = MPNCOV.CovpoolLayer(x3)
    x3 = MPNCOV.SqrtmLayer(x3, 5)
    x3 = MPNCOV.TriuvecLayer(x3)
    x3 = x3.view(x3.size(0), -1)
    x3 = self.fc3(x3)
    x3 = self.bn3(x3)
    x3t = self.act3(x3)
    x3 = self.fc31(x3t)

    x4 = MPNCOV.CovpoolLayer(x4)
    x4 = MPNCOV.SqrtmLayer(x4, 5)
    x4 = MPNCOV.TriuvecLayer(x4)
    x4 = x4.view(x4.size(0), -1)
    x4 = self.fc4(x4)
    x4 = self.bn4(x4)
    x4t = self.act4(x4)
    x4 = self.fc41(x4t)

    return x2, x3, x4, x2t, x3t, x4t


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def train(model, device, train_loader, optimizer, epoch):
  batch_time = AverageMeter() 
  data_time = AverageMeter()
  losses = AverageMeter()

  model.train()

  end = time.time()

  for i, sample in enumerate(train_loader):

    data_time.update(time.time() - end) 

    data, label = sample['data'], sample['label']

    shape = list(data.size())
    # print(shape)
    data = data.reshape(shape[0] * shape[1], *shape[2:])
    label = label.reshape(-1)

    data, label = data.to(device), label.to(device)

    optimizer.zero_grad()

    end = time.time()

    out3, out4, out5, out3fc, out4fc, out5fc = model(data)  #FP

    criterion = nn.CrossEntropyLoss()
    loss3 = criterion(out3, label)
    loss4 = criterion(out4, label)
    loss5 = criterion(out5, label)
    out3fc_de = out3fc.detach()
    out3fc_de = cs_sc(out3fc_de)
    out4fc_de = out4fc.detach()
    out4fc_de = cs_sc(out4fc_de)
    out5fc_de = out5fc.detach()
    out5fc_de = cs_sc(out5fc_de)
    loss3_p = pairedloss(out3fc, out3fc_de)
    loss4_p = pairedloss(out4fc, out4fc_de)
    loss5_p = pairedloss(out5fc, out5fc_de)
    loss = loss3 + loss4 + loss5 + 0.9 * (loss3_p + loss4_p + loss5_p)

    losses.update(loss.item(), data.size(0))

    loss.backward()       #BP
    optimizer.step()

    batch_time.update(time.time() - end) #BATCH TIME = BATCH BP+FP
    end = time.time()

    if i % TRAIN_PRINT_FREQUENCY == 0:

      logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

#Adjust BN estimated value
def adjust_bn_stats(model, device, train_loader):
  model.train()

  with torch.no_grad():
    for sample in train_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      out3, out4, out5, _, _, _ = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
  model.eval()

  test_loss = 0
  correct3 = 0
  correct4 = 0
  correct5 = 0

  with torch.no_grad():
    for sample in eval_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      out3, out4, out5, _, _, _ = model(data)
      pred3 = out3.max(1, keepdim=True)[1]
      correct3 += pred3.eq(label.view_as(pred3)).sum().item()

      pred4 = out4.max(1, keepdim=True)[1]
      correct4 += pred4.eq(label.view_as(pred4)).sum().item()

      pred5 = out5.max(1, keepdim=True)[1]
      correct5 += pred5.eq(label.view_as(pred5)).sum().item()

  accuracy = correct5 / (len(eval_loader.dataset) * 2)
  accuracy3 = correct3 / (len(eval_loader.dataset) * 2)
  accuracy4 = correct4 / (len(eval_loader.dataset) * 2)

  if accuracy > best_acc and epoch > 180:
    best_acc = accuracy
    all_state = {
      'original_state': model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'epoch': epoch
    }
    torch.save(all_state, PARAMS_PATH)
  
  logging.info('-' * 8)
  logging.info('Eval accuracy: s{:.4f},  m{:.4f},  h{:.4f}'.format(accuracy3, accuracy4, accuracy))
  logging.info('Best accuracy:{:.4f}'.format(best_acc))   
  logging.info('-' * 8)
  return best_acc


def svm_class(model, device, train_loader, eval_loader):
  model.eval()

  out_train = []
  out_eval = []
  out_train_y = []
  out_eval_y = []

  with torch.no_grad():
    for sample in train_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      _, _, _, out3, out4, out5 = model(data)
      out = torch.cat((out3, out4, out5), dim=1)
      out_train.append(out)
      out_train_y.append(label)

    for sample in eval_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      _, _, _, out3, out4, out5 = model(data)
      out = torch.cat((out3, out4, out5), dim=1)
      out_eval.append(out)
      out_eval_y.append(label)
  out_train = torch.cat(out_train, dim=0).cpu().numpy()
  out_train_y = torch.cat(out_train_y, dim=0).cpu().numpy()
  out_eval = torch.cat(out_eval, dim=0).cpu().numpy()
  out_eval_y = torch.cat(out_eval_y, dim=0).cpu().numpy()
  clf = svm.SVC(kernel='rbf', C=5, gamma='auto')
  clf.fit(out_train, out_train_y)
  score = clf.score(out_eval, out_eval_y)
  logging.info('svm accuracy:{:.4f}'.format(score))


#Initialization
def initWeights(module):
  if type(module) == nn.Conv2d:
    if module.weight.requires_grad:
      nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

  if type(module) == nn.Linear:
    nn.init.normal_(module.weight.data, mean=0, std=0.01)
    nn.init.constant_(module.bias.data, val=0)

#Data augmentation 
class AugData():
  def __call__(self, sample):
    data, label = sample['data'], sample['label']

    #Rotation
    rot = random.randint(0,3)
    data = np.rot90(data, rot, axes=[1, 2]).copy()
    
    #Mirroring 
    if random.random() < 0.5:
      data = np.flip(data, axis=2).copy()

    new_sample = {'data': data, 'label': label}

    return new_sample


class ToTensor():
  def __call__(self, sample):
    data, label = sample['data'], sample['label']

    data = np.expand_dims(data, axis=1)
    data = data.astype(np.float32)
    # data = data / 255.0

    new_sample = {
      'data': torch.from_numpy(data),
      'label': torch.from_numpy(label).long(),
    }

    return new_sample


class MyDataset(Dataset):
  def __init__(self, DATASET_DIR, transform=None):
    self.transform = transform
    
    self.cover_dir = DATASET_DIR + '/cover'
    self.stego_dir = DATASET_DIR + '/stego'

    self.cover_list = [x.split('/')[-1] for x in glob(self.cover_dir+'/*')]
    assert len(self.cover_list) != 0, "cover_dir is empty"
    
  def __len__(self):
    return len(self.cover_list)

  def __getitem__(self, idx):
    file_index = int(idx)

    cover_path=os.path.join(self.cover_dir,self.cover_list[file_index])
    stego_path=os.path.join(self.stego_dir,self.cover_list[file_index])
    
    cover_data = cv2.imread(cover_path, -1)
    stego_data = cv2.imread(stego_path, -1)

    data = np.stack([cover_data, stego_data])
    label = np.array([0, 1], dtype='int32')

    sample = {'data': data, 'label': label}

    if self.transform:
      sample = self.transform(sample)

    return sample


def setLogger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


def main(args):

  statePath = args.statePath

  device = torch.device("cuda")

  kwargs = {'num_workers': 4, 'pin_memory': True}

  train_transform = transforms.Compose([
    AugData(),
    ToTensor()
  ])

  eval_transform = transforms.Compose([
    ToTensor()
  ])


  TRAIN_DATASET_DIR = args.TRAIN_DIR
  VALID_DATASET_DIR = args.VALID_DIR
  TEST_DATASET_DIR = args.TEST_DIR

  #Log files
  PARAMS_NAME = 'model_params' + args.gpuNum + '.pt'
  LOG_NAME = 'model_log' + args.gpuNum
  print('按gpu序号写入：', PARAMS_NAME, LOG_NAME)
  
  PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
  LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

  setLogger(LOG_PATH, mode='w')

  Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
  
  train_dataset = MyDataset(TRAIN_DATASET_DIR, train_transform)
  valid_dataset = MyDataset(VALID_DATASET_DIR, eval_transform)
  test_dataset = MyDataset(TEST_DATASET_DIR,  eval_transform)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
  valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

  model = Net().to(device)
  model.apply(initWeights)

  params = model.parameters()

  params_wd, params_rest = [], []
  for param_item in params:
      if param_item.requires_grad:
          (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

  param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
                    {'params': params_rest}]

  optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9)

  if statePath:
    logging.info('-' * 8)
    logging.info('Load state_dict in {}'.format(statePath))
    logging.info('-' * 8)

    all_state = torch.load(statePath)

    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    epoch = all_state['epoch']

    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    startEpoch = epoch + 1

  else:
    startEpoch = 1

  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
  best_acc = 0.0

  for epoch in range(startEpoch, EPOCHS + 1):

    train(model, device, train_loader, optimizer, epoch)
    scheduler.step()

    if epoch % EVAL_PRINT_FREQUENCY == 0:
      adjust_bn_stats(model, device, train_loader)
      best_acc = evaluate(model, device, valid_loader, epoch, optimizer, best_acc, PARAMS_PATH)


  logging.info('\nTest set accuracy: \n')

  #Load best network parmater to test    
  all_state = torch.load(PARAMS_PATH)
  original_state = all_state['original_state']
  optimizer_state = all_state['optimizer_state']
  model.load_state_dict(original_state)
  optimizer.load_state_dict(optimizer_state)

  adjust_bn_stats(model, device, train_loader)
  evaluate(model, device, test_loader, epoch, optimizer, best_acc, PARAMS_PATH)
  svm_class(model, device, train_loader, test_loader)


def myParseArgs():
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-TRAIN_DIR',
    '--TRAIN_DIR',
    help='The path to load train_dataset',
    type=str,
    required=True
  )

  parser.add_argument(
    '-VALID_DIR',
    '--VALID_DIR',
    help='The path to load valid_dataset',
    type=str,
    required=True
  )

  parser.add_argument(
    '-TEST_DIR',
    '--TEST_DIR',
    help='The path to load test_dataset',
    type=str,
    required=True
  )

  parser.add_argument(
    '-g',
    '--gpuNum',
    help='Determine which gpu to use',
    type=str,
    choices=['0', '1', '2', '3'],
    required=True
  )

  parser.add_argument(
    '-l',
    '--statePath',
    help='Path for loading model state',
    type=str,
    default=''
  )

  args = parser.parse_args()

  return args


if __name__ == '__main__':
  args = myParseArgs()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
  main(args)


