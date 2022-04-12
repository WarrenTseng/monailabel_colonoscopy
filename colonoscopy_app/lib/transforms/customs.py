import torch
import torch.nn.functional as F
import monai
import monailabel
from monailabel.transform.post import Restored
import matplotlib.pyplot as plt
from skimage.transform import resize

from monai.utils.enums import TransformBackends
from monai.utils import (
    InterpolateMode,
    ensure_tuple_size,
    fall_back_tuple,
)
from monai.config.type_definitions import NdarrayOrTensor
from typing import Any, Hashable, Callable, List, Optional, Sequence, Tuple, Union
from monai.utils.module import look_up_option
from monai.utils.enums import TraceKeys
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.networks.layers import AffineTransform
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.croppad.array import CenterSpatialCrop, SpatialPad
from monai.transforms.inverse import InvertibleTransform

from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.utils import create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    PytorchPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)
from monai.utils.deprecate_utils import deprecated_arg
from monai.utils.enums import TraceKeys
from monai.utils.module import optional_import
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type


class LoadColonoscopyJPGd(monai.transforms.MapTransform):
    def __init__(self, key):
        self.key = key
        
    def __call__(self, inputs):
        if type(self.key) is list or type(self.key) is tuple:
            for k in self.key:
                inputs[k] = plt.imread(inputs[k])
        else:
            inputs[self.key] = plt.imread(inputs[self.key])
        return inputs

class ShapeRecordd(monai.transforms.MapTransform):
    def __init__(self, key, img_key):
        self.key = key
        self.img_key = img_key
    
    def __call__(self, inputs):
        inputs['shape'] = None
        inputs[self.key] = inputs[self.img_key].shape
        return inputs

class Normalized(monai.transforms.MapTransform):
    def __init__(self, key, mean, std):
        self.key = key
        self.mean = mean
        self.std = std

    def __call__(self, inputs):
#         inputs[self.key] = inputs[self.key]/255.
        if type(self.mean) is list or type(self.key) is tuple:
            outputs = torch.zeros_like(inputs[self.key])
            for i in range(len(self.mean)):
                outputs[i] = (inputs[self.key][i]-self.mean[i])/self.std[i]
        else:
            outputs = (inputs[self.key]-self.mean)/self.std
        inputs[self.key] = outputs
        return inputs
    
class Upsamplingd(monai.transforms.MapTransform):
    def __init__(self, key, size_key, mode='nearest'):
        self.key = key
        self.size_key = size_key
        self.mode = mode
        
    def __call__(self, inputs):
        inputs[self.key] = F.upsample(inputs[self.key], size=inputs[self.size_key][1:], mode=self.mode)[0]
        return inputs
    
class ToUint8d(monai.transforms.MapTransform):
    def __init__(self, key):
        self.key = key
        
    def __call__(self, inputs):
        inputs[self.key] = (inputs[self.key]).astype('uint8')
#         inputs[self.key] = np.concatenate([inputs[self.key][..., None] for i in range(3)], axis=-1)
#         plt.imsave('/tmp/test.jpg', inputs[self.key])
#         print(inputs[self.key].max())
#         np.save('/tmp/test2.npy', inputs[self.key])
        return inputs

class PraNetOutd(monai.transforms.MapTransform):
    def __init__(self, key):
        self.key = key
        
    def __call__(self, inputs):
        inputs[self.key] = inputs[self.key][0]
        return inputs

class _PraNetOutd(monai.transforms.MapTransform):
    def __init__(self, key):
        self.key = key
        
    def __call__(self, inputs):
        np.save('/tmp/test.npy', np.concatenate([inputs['label'].cpu().detach().numpy(), inputs['pred'].cpu().detach().numpy()]))
        return inputs    

class Resized(monai.transforms.MapTransform):
    from skimage.transform import resize
    
    def __init__(self, key, size):
        self.key = key
        self.size = size

    def __call__(self, inputs):
        if type(self.key) is list or type(self.key) is tuple:
            for k in self.key:
                print(k, inputs[k].max(), inputs[k].shape)
                if k == 'label':
                    order = 0
                    s = (1,)+self.size[1:]
                    inputs[k] = inputs[k][..., 0].transpose([0, 2, 1])
                else:
                    inputs[k] = inputs[k].astype('uint8')
                    order = 1
                    s = self.size
                
                inputs[k] = resize(inputs[k], s, order=order)
                print('done', k, inputs[k].max(), inputs[k].shape)
        else:
            inputs[self.key] = inputs[self.key].astype('uint8')
            inputs[self.key] = resize(inputs[self.key], self.size) 
        return inputs    
    
    
