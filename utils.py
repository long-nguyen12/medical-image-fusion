# Modifed form timm and swin repo.

""" CUDA / AMP utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
import numpy as np
import torch.nn as nn

try:
    from apex import amp

    has_apex = True
except ImportError:
    amp = None
    has_apex = False

from timm.utils.clip_grad import dispatch_clip_grad


class ApexScalerAccum:
    state_dict_key = "amp"

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        clip_mode="norm",
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                dispatch_clip_grad(
                    amp.master_params(optimizer), clip_grad, mode=clip_mode
                )
            optimizer.step()

    def state_dict(self):
        if "state_dict" in amp.__dict__:
            return amp.state_dict()

    def load_state_dict(self, state_dict):
        if "load_state_dict" in amp.__dict__:
            amp.load_state_dict(state_dict)


class NativeScalerAccum:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        clip_mode="norm",
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(
            torch.stack(self.losses[np.maximum(len(self.losses) - self.num, 0) :])
        )


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
