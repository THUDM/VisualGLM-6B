import os
from PIL import Image
from io import BytesIO
import base64
import re
import argparse
import torch
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin

from .visualglm import VisualGLMModel

def get_infer_setting(gpu_device=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    args = argparse.Namespace(
        fp16=True,
        skip_init=True
    )
    model, args = VisualGLMModel.from_pretrained('visualglm-6b', args)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    return model, tokenizer

def is_chinese(text):
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    return zh_pattern.search(text)

def generate_input(input_text, input_image_prompt, history=[], input_para=None, image_is_encoded=True):
    if not image_is_encoded:
        image = input_image_prompt
    else:
        decoded_image = base64.b64decode(input_image_prompt)
        image = Image.open(BytesIO(decoded_image))

    input_data = {'input_query': input_text, 'input_image': image, 'history': history, 'gen_kwargs': input_para}
    return input_data