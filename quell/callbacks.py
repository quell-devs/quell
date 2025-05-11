import numpy as np
from rich.progress import track
from PIL import Image
import torch
from fastai.callback.core import Callback, CancelBatchException


class NoiseIncludeCallBack(Callback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_batch(self):
        noise = self.yb[0] - self.xb[0]

        self.learn.xb = (self.xb[0],)
        self.learn.yb = (self.yb[0],noise)

class IdentityCallback(Callback):
    '''
    Return noise residual of identity model is zero.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def after_pred(self):
        self.learn.pred = torch.zeros_like(self.pred)

