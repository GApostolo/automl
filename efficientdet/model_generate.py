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
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import logging


from ..efficientdet import inference
from ..efficientdet.model_inspect import ModelInspector
from ..efficientdet.tf2 import infer_lib


def load_model_from_frozen_file(model_name: str = 'efficientdet-d0',
                                saved_model_dir: str = './model/automl/efficientdet/model_save',
                                min_score_thresh: int = 0.5, max_boxes_to_draw: int = 500, batch_size: int = 1,
                                nms_method: str = 'hard', hparams_file_name: str = '/panda_config.yaml'):
    model_dir = saved_model_dir
    hparams = hparams_file_name
    model_config = ModelInspector(model_name=model_name, saved_model_dir=model_dir, batch_size=batch_size,
                                  score_thresh=min_score_thresh, max_output_size=max_boxes_to_draw,
                                  nms_method=nms_method, hparams=hparams)

    driver = inference.ServingDriver(model_name, model_config.batch_size,
                                     model_params=model_config.model_config.as_dict(),
                                     min_score_thresh=min_score_thresh, max_boxes_to_draw=max_boxes_to_draw)
    drive_start_time = time.perf_counter()
    driver.load(model_dir)
    drive_end_time = time.perf_counter()
    print(f"Drive time (seconds): {drive_end_time - drive_start_time:.2f}")
    return driver


def load_model_from_checkpoint_file(model_name: str = 'efficientdet-d0',
                                    saved_model_dir: str = './model/automl/efficientdet/model_save',
                                    min_score_thresh: int = 0.5, max_boxes_to_draw: int = 500, batch_size: int = 1,
                                    nms_method: str = 'hard', hparams_file_name: str = '/panda_config.yaml'):
    model_dir = saved_model_dir
    hparams = hparams_file_name
    model_config = ModelInspector(model_name=model_name, saved_model_dir=model_dir, batch_size=batch_size,
                                  score_thresh=min_score_thresh, max_output_size=max_boxes_to_draw, hparams=hparams,
                                  nms_method=nms_method)

    driver = inference.ServingDriver(model_name, model_config.batch_size,
                                     model_params=model_config.model_config.as_dict(),
                                     min_score_thresh=min_score_thresh, max_boxes_to_draw=max_boxes_to_draw)
    drive_start_time = time.perf_counter()
    driver.ckpt_path = model_dir
    driver.build()
    drive_end_time = time.perf_counter()
    print(f"Drive time (seconds): {drive_end_time - drive_start_time:.2f}")
    return driver


def load_tf2_model_from_frozen_file(model_name: str = 'efficientdet-d0',
                                    saved_model_dir: str = './model/automl/efficientdet/',
                                    min_score_thresh: int = 0.5, max_boxes_to_draw: int = 500, batch_size: int = 1,
                                    nms_method: str = 'hard'):
    model_dir = saved_model_dir
    hparams = saved_model_dir + "/config.yaml"
    model_config = ModelInspector(model_name=model_name, saved_model_dir=model_dir, batch_size=batch_size,
                                  score_thresh=min_score_thresh, max_output_size=max_boxes_to_draw,
                                  nms_method=nms_method, hparams=hparams)
    logger = tf.get_logger()
    logger.setLevel(logging.FATAL)

    real_absl_logger = logging.getLogger("absl")
    real_absl_logger.setLevel(logging.FATAL)

    model_params = model_config.model_config.as_dict()
    driver = infer_lib.ServingDriver.create('_', False, model_dir, model_name, batch_size, False, model_params)
    return driver
