import os.path
import logging
import time
from collections import OrderedDict
import numpy
import torch
from torch._C import parse_ir

from utils import utils_logger
from utils import utils_image as util
from RFDN import RFDN
from RFDNsmall import RFDNsmall
from tqdm import tqdm
import argparse


def train():
    utils_logger.logger_info('AIM-track', log_path='AIM-track.log')
    logger = logging.getLogger('AIM-track')
    testsets = 'DIV2K'
    testset_L = 'DIV2K_valid_LR_bicubic'
    trainset_L = 'DIV2K_train_LR_bicubic'
    #testset_L = 'DIV2K_test_LR_bicubic'
    real_L = 'DIV2K_valid_HR'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    L_folder = os.path.join(testsets, testset_L, 'X4')
    TRAIN_folder = os.path.join(testsets, trainset_L, 'X4')
    E_folder = os.path.join(testsets, testset_L+'_results')
    H_folder = os.path.join(testsets, real_L)
    
    model_path = os.path.join('trained_model', 'RFDN_AIM.pth')
    actual_model = RFDN()
    destilled_model = RFDNsmall()
    weights = torch.load(model_path)
    actual_model.load_state_dict(weights, strict=True)
    actual_model.eval()
    for k, v in actual_model.named_parameters():
        v.requires_grad = False
    actual_model = actual_model.to(device)
    destilled_model = destilled_model.to(device)
    loss = torch.nn.MSELoss()
    lr =0.0005
    optimizer = torch.optim.Adam(destilled_model.parameters(), lr=lr)
    epochs = 1
    for i in range(epochs):
        print("Epoch: ", i)
        for img in tqdm(util.get_image_paths(TRAIN_folder)):
            optimizer.zero_grad()
            img_L = util.imread_uint(img, n_channels=3)
            img_L = util.uint2tensor4(img_L)
            img_L = img_L.to(device)
            actual_out = actual_model.destill_forward(img_L)
            destilled_out = destilled_model.destill_forward(img_L)
            sum_loss = sum([loss(el1, el2) for el1, el2 in zip(actual_out, destilled_out)])
            sum_loss.backward()
            optimizer.step()
        eval(destilled_model, L_folder, H_folder, device)


def eval(model, L_folder, H_folder, device):
    img_SR = []
    psnr = []
    idx = 0
    for img in tqdm(util.get_image_paths(L_folder)):
        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)
        img_E = model(img_L)
        img_E = util.tensor2uint(img_E)
    for img in tqdm(util.get_image_paths(H_folder)):
        img_H = util.imread_uint(img, n_channels=3)
        psnr.append(util.calculate_psnr(img_SR[idx], img_H))
        idx += 1
    del img_SR
    print('------> Average psnr of ({}) is : {:.6f} dB'.format(L_folder, sum(psnr)/len(psnr)))

if __name__ == '__main__':
    train()