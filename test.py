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
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantinize', action='store', default=False, type=bool)
    parser.add_argument('--destilled-model')
    utils_logger.logger_info('AIM-track', log_path='AIM-track.log')
    logger = logging.getLogger('AIM-track')

    # --------------------------------
    # basic settings
    # --------------------------------
    testsets = 'DIV2K'
    testset_L = 'DIV2K_valid_LR_bicubic'
    #testset_L = 'DIV2K_test_LR_bicubic'
    real_L = 'DIV2K_valid_HR'

    if torch.cuda.is_available():
        torch.cuda.current_device()
        torch.cuda.empty_cache()
    #torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --------------------------------
    # read image
    # --------------------------------
    L_folder = os.path.join(testsets, testset_L, 'X4')
    E_folder = os.path.join(testsets, testset_L+'_results')
    H_folder = os.path.join(testsets, real_L)
    util.mkdir(E_folder)

    # --------------------------------
    # load model
    # --------------------------------
    model_path = os.path.join('trained_model', 'RFDN_AIM.pth')
    model = RFDN()
    weights = torch.load(model_path)
    pre_size = sum([weight.element_size() * weight.nelement() for weight in weights.values()])
    print("Size before processing in MBits: ", pre_size / 1000**2)
    # for name in weights.keys():
    #     weights[name] = weights[name].to(torch.float16)
    post_size = sum([weight.element_size() * weight.nelement() for weight in weights.values()])
    print("Size after processing in MBits: ", post_size / 1000**2)
    while True:
        model = RFDN()
        try:
            model.load_state_dict(weights, strict=True)
            break
        except RuntimeError as err:
            if (str(err).startswith('Error(s) in loading state_dict for RFDN')):
                name = str(err).split('"')[1]
                print("deleting", name)
                del weights[str(err).split('"')[1]]
            else:
                return
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    # torch.onnx.export(model, torch.rand(1,3,339,510), 'rfdn.onnx')
    model = model.to(device)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    # --------------------------------
    dynamic = False
    quantize_torch = True
    if quantize_torch:
        seqModel = torch.nn.Sequential(torch.quantization.QuantStub(), model, torch.quantization.DeQuantStub())
        if dynamic:
            model = torch.quantization.quantize_dynamic(
                model,  # the original model
                {torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.InstanceNorm2d, torch.nn.LeakyReLU, torch.nn.ReLU},  # a set of layers to dynamically quantize
                dtype=torch.quint8)  # the target dtype for quantized weights
        else:
            torch.backends.quantized.engine = 'qnnpack'
            seqModel.qconfig = torch.quantization.get_default_qconfig('qnnpack')

            # Fuse the activations to preceding layers, where applicable.
            # This needs to be done manually depending on the model architecture.
            # Common fusions include `conv + relu` and `conv + batchnorm + relu`
            # seqModel = torch.quantization.fuse_modules(seqModel, [])
            

            # Prepare the model for static quantization. This inserts observers in
            # the model thats will observe activation tensors during calibration.
            model_fp32_prepared = torch.quantization.prepare(seqModel)

            # calibrate the prepared model to determine quantization parameters for activations
            # in a real world setting, the calibration would be done with a representative dataset
            for index, img in enumerate(tqdm(util.get_image_paths(L_folder))):
                img_L = util.imread_uint(img, n_channels=3)
                img_L = util.uint2tensor4(img_L)
                img_L = img_L.to(device)
                model_fp32_prepared(img_L)
                if index == 15:
                    break

            # Convert the observed model to a quantized model. This does several things:
            # quantizes the weights, computes and stores the scale and bias value to be
            # used with each activation tensor, and replaces key operators with quantized
            # implementations.
            model_int8 = torch.quantization.convert(model_fp32_prepared)
            model = model_int8

    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(L_folder)
    logger.info(E_folder)
    idx = 0

    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
    else:
        start = time.time()
        end = time.time()

    img_SR = []
    for img in tqdm(util.get_image_paths(L_folder)):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        #logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        if torch.cuda.is_available():
            start.record()
        else:
            start = time.time()
        img_E = model(img_L)
        if torch.cuda.is_available():
            end.record()
        else:
            end = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            test_results['runtime'].append(start.elapsed_time(end))  # milliseconds
        else:
            test_results['runtime'].append((end - start) * 1000)

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)
        img_SR.append(img_E)

        # --------------------------------
        # (3) save results
        # --------------------------------
        # util.imsave(img_E, os.path.join(E_folder, img_name+ext))

    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_folder, ave_runtime))

    # --------------------------------
    # (4) calculate psnr
    # --------------------------------
    psnr = []
    idx = 0
    #H_folder = '/home/lj/EfficientSR-1.5.0/train/dataset/benchmark/DIV2K_valid/HR/'
    for img in tqdm(util.get_image_paths(H_folder)):
        img_H = util.imread_uint(img, n_channels=3)
        psnr.append(util.calculate_psnr(img_SR[idx], img_H))
        idx += 1
    logger.info('------> Average psnr of ({}) is : {:.6f} dB'.format(L_folder, sum(psnr)/len(psnr)))


if __name__ == '__main__':

    main()
