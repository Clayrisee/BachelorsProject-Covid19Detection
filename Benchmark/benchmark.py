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
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def get_convnext_model():
    cfg = edict(yaml.safe_load(open("models/convnext/model-config.yaml", "r")))
    model_kwargs = dict(
    input_size= cfg.model_input,
    backbone=cfg.model,
    output_class=cfg.output_class
    )
    model = InferenceModel(**model_kwargs)
    pretrained_path = os.path.join('models/convnext', cfg.model_file)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    return model

def get_vit_model():
    cfg = edict(yaml.safe_load(open("models/vit/model-config.yaml", "r")))
    model_kwargs = dict(
    input_size= cfg.model_input,
    backbone=cfg.model,
    output_class=cfg.output_class
    )
    model = InferenceModel(**model_kwargs)
    pretrained_path = os.path.join('models/vit', cfg.model_file)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    return model

def get_efficientnet_b0_model():
    cfg = edict(yaml.safe_load(open("models/efficientnet/model-config.yaml", "r")))
    model_kwargs = dict(
    input_size= cfg.model_input,
    backbone=cfg.model,
    output_class=cfg.output_class
    )
    model = InferenceModel(**model_kwargs)
    pretrained_path = os.path.join('models/efficientnet', cfg.model_file)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    return model

def get_prediction_model(benchmark_df:pd.DataFrame, model) -> torch.Tensor:
    list_prediction = list()
    list_conf = list()
    # model = model.to(device)
    for _, row in benchmark_df.iterrows():
        img_path = os.path.join('covid_dataset', row['folder'], "images", f"{row['filename']}.{row['format'].lower()}")
        # print(img_path)
        pil_img = Image.open(img_path).convert('RGB')
        predicts = torch.squeeze(model.predict(pil_img), dim=0)
        pred_label = int(torch.argmax(predicts))
        list_prediction.append(pred_label)
        list_conf.append(predicts[pred_label].detach().float())
    return torch.Tensor(list_prediction), torch.Tensor(list_conf)

if __name__ == "__main__":
    print("--System Started--")
    benchmark_df = pd.read_csv('covid_dataset/test.csv')
    # print(benchmark_df['labels'])
    gt_label = torch.Tensor(benchmark_df["labels"].values)
    # print(gt_label)
    metric = ClassificationMetrics()
    # convnext_model = get_convnext_model()
    vit_model = get_vit_model()
    print("Model Loaded")
    # effnet_b0_model = get_efficientnet_b0_model()
    print("Get Prediction...")
    pred_label, pred_conf = get_prediction_model(benchmark_df, vit_model)
    print("Get Result Metric..")
    result_metric = metric(gt_label, pred_label)
    result_metric['model'] = 'vit'
    print(result_metric)
    result_pred_df = pd.DataFrame({'ground_truth': gt_label, 
    'pred_label':pred_label.cpu().detach().numpy(), 
    'pred_conf':pred_conf.cpu().detach().numpy()})
    result_df = pd.DataFrame(result_metric, index=[0])
    # result_df.to_csv("efficientnet_benchmark_metric_result.csv", index=False)
    # result_pred_df.to_csv("efficientnet_benchmark_result.csv", index=False)

    # result_df.to_csv("convnext_benchmark_metric_result.csv", index=False)
    # result_pred_df.to_csv("convnext_benchmark_result.csv", index=False)

    result_df.to_csv("vit_benchmark_metric_result.csv", index=False)
    result_pred_df.to_csv("vit_benchmark_result.csv", index=False)
    print("--System All Green--")