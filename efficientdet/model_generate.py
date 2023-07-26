# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Tool to inspect a model."""
import pathlib
from typing import Text, Tuple, List

from absl import app
from absl import logging

from ..efficientdet import inference

from ..efficientdet.model_inspect import ModelInspector


def load_model(model_name: str = 'efficientdet-d0', saved_model_dir: str = './model/automl/efficientdet/model_save',
               min_score_thresh: int = 0.5, max_boxes_to_draw: int = 500, batch_size: int = 1,
               nms_method: str = 'hard'):
    model_dir = saved_model_dir+"/"+model_name
    model_config = ModelInspector(model_name=model_name, saved_model_dir=model_dir, batch_size=batch_size,
                                  score_thresh=min_score_thresh, max_output_size=max_boxes_to_draw,
                                  nms_method=nms_method)

    driver = inference.ServingDriver(model_name, model_config.batch_size,
                                     model_params=model_config.model_config.as_dict(),
                                     min_score_thresh=min_score_thresh, max_boxes_to_draw=max_boxes_to_draw)
    driver.load(model_dir)
    return driver
