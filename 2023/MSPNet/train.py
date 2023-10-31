#!/usr/bin/env python3
from sklearn import svm
import os
import argparse
from pathlib import Path
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from LD_loss.PD import LD_loss
from LD_loss.PD import tramsform
from dataset.dataset import AugData
from dataset.dataset import ToTensor
from dataset.dataset import MyDataset
from models.model import Model

IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2

EPOCHS = 200
LR = 0.01

WEIGHT_DECAY = 5e-4

TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [80, 140, 180]

OUTPUT_PATH = Path(__file__).stem


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

    out1, out2, out3, fv_1, fv_2, fv_3 = model(data)  #FP

    criterion = nn.CrossEntropyLoss()
    loss_cls1 = criterion(out1, label)
    loss_cls2 = criterion(out2, label)
    loss_cls3 = criterion(out3, label)
    fv_1_de = fv_1.detach()
    fv_1_de = tramsform(fv_1_de)
    fv_2_de = fv_2.detach()
    fv_2_de = tramsform(fv_2_de)
    fv_3_de = fv_3.detach()
    fv_3_de = tramsform(fv_3_de)
    loss_ld1 = LD_loss(fv_1, fv_1_de)
    loss_ld2 = LD_loss(fv_2, fv_2_de)
    loss_ld3 = LD_loss(fv_3, fv_3_de)
    loss = loss_cls1 + loss_cls2 + loss_cls3 + 0.9 * (loss_ld1 + loss_ld2 + loss_ld3)

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

      out1, out2, out3, _, _, _ = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH):
  model.eval()

  correct1 = 0
  correct2 = 0
  correct3 = 0

  with torch.no_grad():
    for sample in eval_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      out1, out2, out3, _, _, _ = model(data)
      pred1 = out1.max(1, keepdim=True)[1]
      correct1 += pred1.eq(label.view_as(pred1)).sum().item()

      pred4 = out2.max(1, keepdim=True)[1]
      correct2 += pred4.eq(label.view_as(pred4)).sum().item()

      pred5 = out3.max(1, keepdim=True)[1]
      correct3 += pred5.eq(label.view_as(pred5)).sum().item()

  accuracy1 = correct1 / (len(eval_loader.dataset) * 2)
  accuracy2 = correct2 / (len(eval_loader.dataset) * 2)
  accuracy3 = correct3 / (len(eval_loader.dataset) * 2)

  if accuracy3 > best_acc and epoch > 180:
    best_acc = accuracy3
    all_state = {
      'original_state': model.state_dict(),
      'optimizer_state': optimizer.state_dict(),
      'epoch': epoch
    }
    torch.save(all_state, PARAMS_PATH)
  
  logging.info('-' * 8)
  logging.info('Eval accuracy: s{:.4f},  m{:.4f},  h{:.4f}'.format(accuracy1, accuracy2, accuracy3))
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

      _, _, _, out1, out2, out3 = model(data)
      out = torch.cat((out1, out2, out3), dim=1)
      out_train.append(out)
      out_train_y.append(label)

    for sample in eval_loader:
      data, label = sample['data'], sample['label']

      shape = list(data.size())
      data = data.reshape(shape[0] * shape[1], *shape[2:])
      label = label.reshape(-1)

      data, label = data.to(device), label.to(device)

      _, _, _, out1, out2, out3 = model(data)
      out = torch.cat((out1, out2, out3), dim=1)
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
  print(PARAMS_NAME, LOG_NAME)
  
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

  model = Model().to(device)
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


