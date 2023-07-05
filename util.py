from functools import reduce
from torch.nn import ModuleList
import torch.nn.functional as F
import copy
import math
import torch
import numpy as np
import csv

# ModuleList: 목록에 하위 모듈을 보관(인덱스로 접근 가능)

def clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0

def load_csv(file_path):
  print(f'Load Data | file path: {file_path}')
  with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    lines = []
    for line in csv_reader:
      line[0] = line[0].replace(';','')
      lines.append(line)
  print(f'Load Complete | file path: {file_path}')

  return lines