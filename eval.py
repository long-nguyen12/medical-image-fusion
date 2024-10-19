# Evaluation Metrics and get results

# Author: Reacher Z., last modify Nov. 26, 2022

"""
Change log:
- Reacher: file created, implement PSNR, SSIM, NMI, MI
"""

import numpy as np
import sklearn.metrics as skm
import torch
from skimage.metrics import peak_signal_noise_ratio, normalized_mutual_information
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import sklearn.metrics as skm
from scipy.signal import convolve2d
import math
#from TMQI import TMQI, TMQIr

def psnr(img_pred: torch.Tensor, img_true: torch.Tensor):
    """
    To compute PeakSignalNoiseRatio
    Return: float
    """
    peakSignalNoiseRatio = PeakSignalNoiseRatio(data_range=1.0)
    return peakSignalNoiseRatio(img_pred, img_true).item()


def ssim(img_pred: torch.Tensor, img_true: torch.Tensor):
    """
    To compute PeakSignalNoiseRatio
    Input: [N, C, H, W] shape
    Return: float
    """
    img_pred = img_pred.unsqueeze(0).unsqueeze(0)
    img_true = img_true.unsqueeze(0).unsqueeze(0)
    structuralSimilarityIndexMeasure = StructuralSimilarityIndexMeasure(data_range=1.0)
    return structuralSimilarityIndexMeasure(img_pred, img_true).item()


def nmi(img_pred: torch.Tensor, img_true: torch.Tensor):
    """
    normalized mutual information (NMI)
    Return: float
    """
    img_pred_np = np.array(img_pred.squeeze())
    img_true_np = np.array(img_true.squeeze())
    nor_mi = normalized_mutual_information(img_pred_np, img_true_np)
    return nor_mi


def mutual_information(img_pred: torch.Tensor, img_true: torch.Tensor):
    """
    Mutual Information:
    I(A,B) = H(A) + H(B) - H(A,B)
    H(A)= -sum p(a_i) * log p(a_i)
    Mutual information is a measure of image matching, that does not require the signal
    to be the same in the two images. It is a measure of how well you can predict the signal
    in the second image, given the signal intensity in the first.
    Return: float
    """
    img_pred_uint8 = (np.array(img_pred.squeeze()) * 255).astype(np.uint8).flatten()
    img_true_uint8 = (np.array(img_true.squeeze()) * 255).astype(np.uint8).flatten()
    size = img_true_uint8.shape[-1]
    pa = np.histogram(img_pred_uint8, 256, (0, 255))[0] / size
    pb = np.histogram(img_true_uint8, 256, (0, 255))[0] / size
    ha = -np.sum(pa * np.log(pa + 1e-20))
    hb = -np.sum(pb * np.log(pb + 1e-20))

    pab = (np.histogram2d(img_pred_uint8, img_true_uint8, 256, [[0, 255], [0, 255]])[0]) / size
    hab = -np.sum(pab * np.log(pab + 1e-20))
    mi = ha + hb - hab
    # hist_2d, x_edges, y_edges = np.histogram2d(img_pred.numpy().ravel(), img_true.numpy().ravel(), bins=256)
    # pxy = hist_2d / float(np.sum(hist_2d))
    # px = np.sum(pxy, axis=1) # marginal for x over y
    # py = np.sum(pxy, axis=0) # marginal for y over x
    # px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # # Now we can do the calculation using the pxy, px_py 2D arrays
    # nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    # return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi

def mi2(x, y):
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)
    return skm.mutual_info_score(x, y)

def Qabf(image_F, image_A, image_B):
    gA, aA = Qabf_getArray(image_A)
    gB, aB = Qabf_getArray(image_B)
    gF, aF = Qabf_getArray(image_F)
    QAF = Qabf_getQabf(aA, gA, aF, gF)
    QBF = Qabf_getQabf(aB, gB, aF, gF)

    # 计算QABF
    deno = np.sum(gA + gB)
    nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
    return nume / deno

def Qabf_getArray(img):
    # Sobel Operator Sobel算子
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    SAx = convolve2d(img, h3, mode='same')
    SAy = convolve2d(img, h1, mode='same')
    gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
    aA = np.zeros_like(img)
    aA[SAx == 0] = math.pi / 2
    aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
    return gA, aA

def Qabf_getQabf(aA, gA, aF, gF):
    L = 1
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8
    GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
    GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
    GAF[gA == gF] = gF[gA == gF]
    GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
    AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
    QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
    QAF = QgAF* QaAF
    return QAF

