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
import logging

import torch
import monai
from monai.handlers import TensorBoardImageHandler, from_engine
from monai.inferers import SlidingWindowInferer, SimpleInferer
from monai.losses import DiceCELoss
from monai.transforms import (
    Activationsd,
    AddChanneld,
    AsDiscreted,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
#     Resized,
    ScaleIntensityRanged,
    SelectItemsd,
    ToNumpyd,
    ToTensord,
)
from lib.transforms.customs import (
    LoadColonoscopyJPGd,
    ShapeRecordd,
    Normalized,
    Upsamplingd,
    Resized,
    PraNetOutd,
    _PraNetOutd
)
from monailabel.deepedit.multilabel.transforms import NormalizeLabelsInDatasetd
from monailabel.tasks.train.basic_train import BasicTrainTask, Context
from monailabel.tasks.train.utils import region_wise_metrics

logger = logging.getLogger(__name__)


class Segmentation(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        spatial_size=(352, 352),  # Depends on original width, height and depth of the training images
        num_samples=4,
        description="Train 2D Segmentation model",
        **kwargs,
    ):
        self._network = network
        self.spatial_size = spatial_size
        self.num_samples = num_samples
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), lr=1e-3)

    def loss_function(self, context: Context):
        return DiceCELoss(to_onehot_y=False, sigmoid=True)

    def lr_scheduler_handler(self, context: Context):
        return None

    def train_data_loader(self, context, num_workers=0, shuffle=False):
        return super().train_data_loader(context, num_workers, True)

    def train_pre_transforms(self, context: Context):
        return [
            LoadColonoscopyJPGd(key="image"),
            LoadImaged(keys="label"),
#             NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),
            AddChanneld(keys=("label")),
            monai.transforms.AsChannelFirstd(('image')),
            Resized(('image', 'label'), (3,)+self.spatial_size),
            RandFlipd(keys=("image", "label"), spatial_axis=[0], prob=0.50),
            RandFlipd(keys=("image", "label"), spatial_axis=[1], prob=0.50),
            ToTensord(keys=("image", "label"), dtype=torch.float32),
            Normalized('image',
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ]

    def train_post_transforms(self, context: Context):
        return [
            PraNetOutd(key="pred"),
            Activationsd(keys="pred", sigmoid=True),
#             AsDiscreted(
#                 keys=("pred", "label"),
#                 argmax=(False, False),
#                 to_onehot=(False, False),
#                 n_classes=len(self._labels),
#             ),
            _PraNetOutd(key=("pred", "label")),
        ]

    def val_pre_transforms(self, context: Context):
        return [
            LoadColonoscopyJPGd(key=("image", "label")),
#             NormalizeLabelsInDatasetd(keys="label", label_names=self._labels),
            AddChanneld(keys=("label")),
            monai.transforms.AsChannelFirstd(('image')),
            Resized(('image', 'label'), (1,)+self.spatial_size),
            ToTensord(keys=("image", "label"), dtype=torch.float32),
            Normalized('image',
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ]

    def val_inferer(self, context: Context):
        return SimpleInferer()

    def train_key_metric(self, context: Context):
        return region_wise_metrics(self._labels, self.TRAIN_KEY_METRIC, "train")

    def val_key_metric(self, context: Context):
        return region_wise_metrics(self._labels, self.VAL_KEY_METRIC, "val")

    def train_handlers(self, context: Context):
        handlers = super().train_handlers(context)
        if context.local_rank == 0:
            handlers.append(
                TensorBoardImageHandler(
                    log_dir=context.events_dir,
                    batch_transform=from_engine(["image", "label"]),
                    output_transform=from_engine(["pred"]),
                    interval=20,
                    epoch_level=True,
                )
            )
        return handlers
