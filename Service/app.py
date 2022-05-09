import yaml
from easydict import EasyDict as edict
from inference_module.inference_models import InferenceModel
import os
import torch
from PIL import Image
import os
from flask import Flask, request
from werkzeug.exceptions import BadRequest, InternalServerError


app = Flask(__name__)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
cfg = edict(yaml.safe_load(open("models/model-config.yaml", "r")))
model_kwargs = dict(
    input_size= cfg.model_input,
    backbone=cfg.model,
    output_class=cfg.output_class
    )
model = InferenceModel(**model_kwargs)
pretrained_path = os.path.join('models', cfg.model_file)
model.load_state_dict(torch.load(pretrained_path, map_location=device))

@app.errorhandler(BadRequest)
def bad_request_handler(error):
    return {
        "error": error.description
    }, 400


@app.errorhandler(InternalServerError)
def internal_server_error_handler(error):
    return {
        "error": error.description
    }, 500

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return {
            "error": "Image is required"
        }, 400
    
    supported_mimetypes = ["image/jpeg", "image/png"]
    mimetype = file.content_type
    if mimetype not in supported_mimetypes:
        return {
            "error": "Unsupported image type"
        }, 415
    img = Image.open(file).convert('RGB')
    result_dict, pred_label = model.predict(img)
    return {
        "result": pred_label,
        "confidence": result_dict
    }, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8989")
