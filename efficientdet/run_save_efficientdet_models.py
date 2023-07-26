"""
Save EffcientDet models
"""
import argparse
import pathlib
import os

from model.automl.efficientdet.model_inspect import ModelInspector

#todo:fix this script
def main():
    args = argparse.ArgumentParser()
    args.add_argument("--max_boxes_to_draw", type=int, default=500, help="Maximum number of detections")
    args.add_argument("--batch_size", type=int, default=1, help="batch size")
    args.add_argument("--min_score_thresh", type=float, default=0.5, help="confidence threshold")
    args = args.parse_args()
    runmode = "saved_model"
    model_name_core = 'efficientdet-d'
    ckpt_path_core = ".\\efficientdet-d"
    hparams = "image_size=3840x2160"
    saved_model = ".\\model_save\\"
    max_boxes_to_draw = args.max_boxes_to_draw
    for network_index in range(8):
        model_name = model_name_core + str(network_index)
        ckpt_path = ckpt_path_core + str(network_index)
        saved_model_dir = saved_model + model_name



if __name__ == "__main__":
    main()
