import sys
import amplify
import torch
import os

def model_fn(model_dir):
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    print(f"Listing files in the current directory: {current_dir}")
    for root, dirs, files in os.walk(current_dir):
        for name in dirs:
            print(f"Directory: {os.path.join(root, name)}")
        for name in files:
            print(f"File: {os.path.join(root, name)}")
    
    print(f"Listing files in the model directory: {model_dir}")
    for root, dirs, files in os.walk(model_dir):
        for name in dirs:
            print(f"Directory: {os.path.join(root, name)}")
        for name in files:
            print(f"File: {os.path.join(root, name)}")

    amplify_model_path = f"{model_dir}/model.safetensors"
    config_path = f"/opt/ml/model/code/conf/config.yaml"
    
    model, tokenizer = amplify.AMPLIFY.load(amplify_model_path, config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return model, tokenizer, device


def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        import json
        data = json.loads(request_body)
        sequence = data['sequence']
        return sequence
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model_artifacts):
    model, tokenizer, device = model_artifacts
    predictor = amplify.inference.Predictor(model, tokenizer, device=device)
    logits = predictor.logits(input_data)
    return logits

def output_fn(prediction, accept='application/json'):
    if accept == 'application/json':
        import json
        return json.dumps({'logits': prediction.tolist()}), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")