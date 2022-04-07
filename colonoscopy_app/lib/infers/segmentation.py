# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Sequence
import torch
import monai
from monai.inferers import SlidingWindowInferer, SimpleInferer
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    EnsureTyped,
    LoadImaged,
    Orientationd,
#     Resized,
    ScaleIntensityRanged,
    SqueezeDimd,
    ToNumpyd,
)
from lib.transforms.customs import (
    LoadColonoscopyJPGd,
    ShapeRecordd,
    Normalized,
    Upsamplingd,
    Resized,
    ToUint8d
)
from monailabel.interfaces.tasks.infer import InferTask, InferType
from monailabel.transform.post import Restored


class Segmentation(InferTask):
    """
    This provides Inference Engine for pre-trained segmentation (UNet) model over MSD Dataset.
    """

    def __init__(
        self,
        path,
        network=None,
        type=InferType.SEGMENTATION,
        labels=None,
        dimension=2,
        description="A pre-trained model for 2D segmentation over Colonoscopy Images",
        **kwargs,
    ):
        super().__init__(
            path=path,
            network=network,
            type=type,
            labels=labels,
            dimension=dimension,
            description=description,
            **kwargs,
        )

    def pre_transforms(self, data=None) -> Sequence[Callable]:
        return [
            LoadColonoscopyJPGd('image'),
            monai.transforms.AsChannelFirstd('image'),
            ShapeRecordd('shape', 'image'),
            Resized('image', [3, 352, 352]),
            monai.transforms.ToTensord('image', torch.float32),
            Normalized('image',
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ]

    def inferer(self, data=None) -> Callable:
        return SimpleInferer()

    def post_transforms(self, data=None) -> Sequence[Callable]:
        largest_cc = False if not data else data.get("largest_cc", False)
        applied_labels = list(self.labels.values()) if isinstance(self.labels, dict) else self.labels
        t = [
#              monai.transforms.ToTensord('pred', torch.float32),
             Upsamplingd(key="pred", size_key='shape'),
             Activationsd(keys="pred", sigmoid=True),
#              AsDiscreted(keys="pred", argmax=False),
             SqueezeDimd(keys="pred", dim=0),
             ToNumpyd(keys="pred"),
             ToUint8d(key="pred")
        ]
        return t
