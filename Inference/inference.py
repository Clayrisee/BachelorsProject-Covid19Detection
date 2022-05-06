"""
this folder contains example of inference module
"""

import yaml
from easydict import EasyDict as edict
from inference_module.inference_models import InferenceModel
import os
import torch
from PIL import Image


cfg = edict(yaml.safe_load(open("models/model-config.yaml", "r")))

model_kwargs = dict(
    input_size= cfg.model_input,
    backbone=cfg.model,
    output_class=cfg.output_class
)
model = InferenceModel(**model_kwargs)
pretrained_path = os.path.join('models', cfg.model_file)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model.load_state_dict(torch.load(pretrained_path, map_location=device))

labels_dict = {
    0: 'Covid',
    1: 'Lung Opacity',
    2: 'Normal',
    3: 'Viral Pneumonia'
}
example_dir = 'example_input'
outputs = list()

for img in os.listdir(example_dir):
    img_path = os.path.join(example_dir, img)
    pil_img = Image.open(img_path).convert('RGB')
    predicts = torch.squeeze(model.predict(pil_img), dim=0)
    # print(predicts)
    # print(torch.argmax(predicts))
    idx = int(torch.argmax(predicts))
    label = labels_dict[idx]
    conf = predicts[idx].detach().float()
    final_output = {
        'img_file':img,
        'label': label,
        'conf': float(conf)
    }
    outputs.append(final_output)
print("List Output")
print(outputs)