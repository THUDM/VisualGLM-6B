import os
from PIL import Image
from io import BytesIO
import base64
import re
import argparse
import torch
from transformers import AutoTokenizer
import deepspeed
from sat.model.mixins import CachedAutoregressiveMixin
from sat import get_tokenizer, get_args

from .visualglm import VisualGLMModel


def get_default_args():
    args=argparse.Namespace(
        fp16=True,
        skip_init=True)
    return args 


def set_default_dist(args, rank, start_port=18000):
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', str(start_port + rank))
    init_method += master_ip + ':' + master_port
    args.world_size, args.rank = 1, 0
    torch.distributed.init_process_group(
            backend='nccl',
            world_size=args.world_size, rank=args.rank, init_method=init_method)
    deepspeed.init_distributed(
            dist_backend='nccl',
            world_size=args.world_size, rank=args.rank, init_method=init_method)


def get_infer_setting(gpu_device=0):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    args = get_default_args()
    model, args = VisualGLMModel.from_pretrained('visualglm-6b', args)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, local_files_only=True)
    tokenizer = get_tokenizer(outer_tokenizer=tokenizer)
    return model, tokenizer   


# for distributed infer
def get_infer_setting2(gpu_device):
    args = get_default_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    set_default_dist(args, gpu_device, 17000)
    model, args = VisualGLMModel.from_pretrained('visualglm-6b', args)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    model.eval()
    return model    


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