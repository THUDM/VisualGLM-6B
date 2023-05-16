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
from sat import get_tokenizer

from .visualglm import VisualGLMModel


def get_default_args():
    args = argparse.Namespace(
        force_inference=True,
        inner_hidden_size=None,
        hidden_size_per_attention_head=None,
        checkpoint_activations=True,
        checkpoint_num_layers=1,
        layernorm_order='pre',
        model_parallel_size=1,
        world_size=1,
        rank=0,
        skip_init=True,
        use_gpu_initialization=True,
        deepspeed=None,
        mode='inference',
        fp16=True,
        local_rank=0,
        pretrain=False        
    )
    args.image_length = 32
    args.ptuning_on_chatglm = False
    args.lora_on_chatglm = True
    args.lora_on_eva2 = False
    args.full_finetune_eva2 = False
    args.tune_norm_bias_chatglm = False
    args.tune_norm_bias_eva2 = False
    args.eva2 = False
    args.eva_clip_vit_lora = True
    args.eva_clip_qformer_lora = False
    args.itc = False   
    return args 


def set_default_dist(args, rank):
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    master_port = os.getenv('MASTER_PORT', str(17000 + rank))
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
            backend='nccl',
            world_size=args.world_size, rank=args.rank, init_method=init_method)
    deepspeed.init_distributed(
            dist_backend='nccl',
            world_size=args.world_size, rank=args.rank, init_method=init_method)


def get_infer_setting(gpu_device):
    args = get_default_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
    set_default_dist(args, 0)
    model, args = VisualGLMModel.from_pretrained('visualglm-6b', args)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, local_files_only=True)
    tokenizer = get_tokenizer(outer_tokenizer=tokenizer)
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


def get_para(input_para=None):
    max_length, num_beams, top_p, temperature = input_para.get('max_length', 512), input_para.get('num_beams', 5), input_para.get('top_p', 0.7), input_para.get('temperature', 0.95)
    return max_length, num_beams, top_p, temperature
