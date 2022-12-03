"""
init benchmark.py
"""
import yaml
from easydict import EasyDict as edict
from benchmark_module.inference_models import InferenceModel
import os
import torch
from PIL import Image
from benchmark_module.metrics import ClassificationMetrics
import pandas as pd
import argparse
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description="Classification benchmark script")
    parser.add_argument("--model_dir", required=True, type=str,
                        help="path to model directory")
    parser.add_argument("--dataset_dir", required=True, type=str,
                        help="path to dataset directory")
    args = parser.parse_args()
    return args


def get_model(model_dir: str):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    cfg_path = os.path.join(model_dir, "model-config.yaml")
    cfg = edict(yaml.safe_load(open(cfg_path, "r")))
    model_kwargs = dict(
    input_size= cfg.model_input,
    backbone=cfg.model,
    output_class=cfg.output_class,
    device = device
    )
    model = InferenceModel(**model_kwargs)
    pretrained_path = os.path.join(model_dir, cfg.model_file)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    return model


def get_prediction_model(benchmark_df:pd.DataFrame, model):
    list_prediction = list()
    list_conf = list()
    for _, row in benchmark_df.iterrows():
        img_path = os.path.join('covid_dataset', row['folder'], "images", f"{row['filename']}.{row['format'].lower()}")
        pil_img = Image.open(img_path).convert('RGB')
        predicts = torch.squeeze(model.predict(pil_img), dim=0)
        pred_label = int(torch.argmax(predicts))
        list_prediction.append(pred_label)
        list_conf.append(predicts[pred_label].detach().float())
    return torch.Tensor(list_prediction), torch.Tensor(list_conf)


def get_dataset(dataset_path: str):
    benchmark_df = pd.read_csv(os.path.join(dataset_path, "test.csv"))
    gt_label = torch.Tensor(benchmark_df["labels"].values)
    return benchmark_df, gt_label


def classification_report(y_true: torch.Tensor, y_pred: torch.Tensor) -> pd.DataFrame:
    cls_metrics = ClassificationMetrics()
    result_dict = cls_metrics(y_true, y_pred)
    result_df = pd.DataFrame(result_dict, index=[0])
    print(result_df)
    return result_df


if __name__ == "__main__":
    args = parse_args()
    benchmark_df, gt_label = get_dataset(args.dataset_dir)
    model = get_model(args.model_dir)
    pred_label, pred_conf = get_prediction_model(benchmark_df, model)
    cls_result_df = classification_report(gt_label, pred_label)
    cls_result_df.to_csv("classification_metrics_report.csv", index=False)
    result_pred_df = pd.DataFrame({
    'pred_label':pred_label.detach().cpu().numpy(), 
    'pred_conf':pred_conf.detach().cpu().numpy()})
    # print(result_pred_df.head())
    final_benchmark_df = pd.concat([benchmark_df, result_pred_df], axis=1)
    final_benchmark_df.to_csv("final_result_pred.csv", index=False)