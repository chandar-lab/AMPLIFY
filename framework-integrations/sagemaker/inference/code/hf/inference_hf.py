import logging
import os
import json  
from transformers import AutoModel, AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO)


def model_fn(model_dir):
    """
    This function loads the AMPLIFY model and tokenizer.
    The model is moved to the GPU to support Flash Attention.
    """
    logging.info("[custom] model_fn: Starting the model loading process...")

    try:
        model_id = os.getenv('AMPLIFY_MODEL_ID', 'chandar-lab/AMPLIFY_120M')
        logging.info(f"[custom] model_fn: Model id is {model_id}")

        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        logging.info(f"[custom] model_fn: Successfully loaded the model: {model}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logging.info(f"[custom] model_fn: Successfully loaded the tokenizer: {tokenizer}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        logging.info(f"[custom] model_fn: Moved model to {device} device")

        return model, tokenizer, device

    except Exception as e:
        
        logging.error(f"[custom] model_fn: Error occurred while loading the model and tokenizer: {str(e)}", exc_info=True)
        raise e  


def input_fn(request_body, content_type='application/json'):
    """
    Pre-processes the input data. Assumes the input is JSON.
    The input should contain a protein sequence.
    """
    logging.info("input_fn: Received input")
    if content_type == 'application/json':
        input_data = json.loads(request_body)  
        sequence = input_data['sequence']
        return sequence
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_artifacts):
    """
    Tokenizes the input protein sequence and runs inference on the model.
    The model is already loaded on the GPU for inference.
    """
    logging.info("predict_fn: Running inference")
    model, tokenizer, device = model_artifacts

    inputs = tokenizer.encode(input_data, return_tensors="pt")

    inputs = inputs.to(device)

    with torch.no_grad():
        output = model(inputs)

    return output


def output_fn(prediction, accept='application/json'):
    """
    Post-processes the output, returning the model's predictions.
    Converts the output to a JSON-serializable format.
    """
    logging.info("output_fn: Formatting output")
    if accept == 'application/json':
        if hasattr(prediction, 'logits'):
            output = prediction.logits
        else:
            raise ValueError(f"Unknown prediction format: {type(prediction)}")

        return json.dumps({"output": output.tolist()}), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")

