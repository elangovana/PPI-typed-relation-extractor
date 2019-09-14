import tempfile

import torch
from PIL import Image
from torchvision import transforms, models
import json

from algorithms.TrainInferencePipeline import TrainInferencePipeline


def input_fn(request_body, request_content_type):
    """An input_fn that processes the request body to a tensor"""
    if request_content_type == 'application/json':

        with tempfile.NamedTemporaryFile("w+b") as f:
            f.write(request_body)
            f.seek(0)
            result = _pre_process_image(f)

        return result
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.
        raise "Unsupported content type {}".format(request_content_type)


def model_fn(model_dir):
    model = TrainInferencePipeline.load(model_dir)
    return model


def predict_fn(input_data, model):
    """Predict using input and model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model(input_data)


def output_fn(prediction, content_type):
    """Return prediction"""
    return json.dumps(prediction.tolist())


def _pre_process_image(image_fp):
    # pre-process data
    image = Image.open(image_fp)
    # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
    transform_pipeline = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
    img = transform_pipeline(image)
    # Add batch [N, C, H, W]
    img_tensor = img.unsqueeze(0)
    return img_tensor
