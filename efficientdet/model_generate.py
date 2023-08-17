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
import time
from typing import Text, Tuple, List

from absl import app
from absl import logging

from ..efficientdet import inference

from ..efficientdet.model_inspect import ModelInspector


def load_model_from_frozen_file(model_name: str = 'efficientdet-d0',
                                saved_model_dir: str = './model/automl/efficientdet/model_save',
                                min_score_thresh: int = 0.5, max_boxes_to_draw: int = 500, batch_size: int = 1,
                                nms_method: str = 'hard'):
    model_dir = saved_model_dir + "/" + model_name + "/" + model_name + "_frozen.pb"
    model_config = ModelInspector(model_name=model_name, saved_model_dir=model_dir, batch_size=batch_size,
                                  score_thresh=min_score_thresh, max_output_size=max_boxes_to_draw,
                                  nms_method=nms_method)

    driver = inference.ServingDriver(model_name, model_config.batch_size,
                                     model_params=model_config.model_config.as_dict(),
                                     min_score_thresh=min_score_thresh, max_boxes_to_draw=max_boxes_to_draw)
    drive_start_time = time.perf_counter()
    driver.load(model_dir)
    drive_end_time = time.perf_counter()
    print(f"Drive time (seconds): {drive_end_time - drive_start_time:.2f}")
    return driver


def load_model_from_checkpoint_file(model_name: str = 'efficientdet-d0',
                                    saved_model_dir: str = './model/automl/efficientdet/',
                                    min_score_thresh: int = 0.5, max_boxes_to_draw: int = 500, batch_size: int = 1,
                                    nms_method: str = 'hard',
                                    hparams: str = './model/automl/efficientdet/panda_config.yaml'):
    model_dir = saved_model_dir + "/" + model_name + '-finetune'
    model_config = ModelInspector(model_name=model_name, saved_model_dir=model_dir, batch_size=batch_size,
                                  score_thresh=min_score_thresh, max_output_size=max_boxes_to_draw, hparams=hparams,
                                  nms_method=nms_method)

    driver = inference.InferenceDriver(model_name, model_dir, model_params=model_config.model_config.as_dict())
    return driver
