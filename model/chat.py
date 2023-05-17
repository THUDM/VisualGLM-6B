# -*- encoding: utf-8 -*-
'''
@File    :   chat.py
@Time    :   2023/05/08 19:10:08
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

import os
import sys
import re
from functools import partial
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import requests
from PIL import Image
from io import BytesIO

import torch
from sat.generation.autoregressive_sampling import filling_sequence, BaseStrategy

from .blip2 import BlipImageEvalProcessor

def get_masks_and_position_ids_glm(seq, mask_position, context_length):
    '''GLM model, different from GPT.
    Args:
        seq: torch.IntTensor, [seq_len]
        mask_position: int, the position of the masked place.
        context_length: int, the length of context.
    Returns:
        tokens: torch.IntTensor, [1, seq_len]
        attention_mask: torch.FloatTensor, [1, seq_len, seq_len]
        position_ids: torch.IntTensor, [2, seq_len]
    '''
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., :context_length] = 1
    attention_mask.unsqueeze_(1)

    # 2D position ids
    position_ids = torch.zeros(2, len(seq), device=tokens.device, dtype=torch.long)
    torch.arange(0, context_length, out=position_ids[0, :context_length])
    position_ids[0, context_length:] = mask_position
    torch.arange(1, len(seq) - context_length + 1, out=position_ids[1, context_length:])

    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids

def process_response(response):
    response = response.strip()
    response = response.replace("[[训练时间]]", "2023年")
    punkts = [
        [",", "，"],
        ["!", "！"],
        [":", "："],
        [";", "；"],
        ["\?", "？"],
    ]
    for item in punkts:
        response = re.sub(r"([\u4e00-\u9fff])%s" % item[0], r"\1%s" % item[1], response)
        response = re.sub(r"%s([\u4e00-\u9fff])" % item[0], r"%s\1" % item[1], response)
    return response

def process_image(text, image=None):
    '''Process image in text.
    Args:
        text: str, text.
        image: Optional, image path / url / PIL image.
    '''
    image_position = text.rfind("<img>") + 5
    # extract path from <img></img> using re
    image_path = re.findall(r"<img>(.*?)</img>", text)
    image_path = image_path[-1] if image_path[-1] else None
    if image_path is not None:
        assert image is None, "image and image_path cannot be both not None."
        text = text.replace(image_path, "")
        image_path = image_path.strip()
        # url
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(response.content))
        # local path
        else:
            image = Image.open(image_path)
    if image is not None and isinstance(image, Image.Image):
        processor = BlipImageEvalProcessor(224)
        image = processor(image.convert('RGB'))
        image = image.unsqueeze(0)
    return text, image_position, image


def chat(image_path, model, tokenizer, 
        query: str, history: List[Tuple[str, str]] = None, image: Image = None,
        max_length: int = 1024, top_p=0.7, top_k=30, temperature=0.95, repetition_penalty=1.2,
        invalid_slices=[], english=False
        ):
    if not history:
        history = []
    if image_path:
        prompt = "<img>{}</img>".format(image_path if image_path else "")
    else:
        prompt = "<img></img>"
    if english:
        for i, (old_query, response) in enumerate(history):
            prompt += "Q:{}\nA:{}\n".format(old_query, response)
        prompt += "Q:{}\nA:".format(query)
    else:
        for i, (old_query, response) in enumerate(history):
            prompt += "问：{}\n答：{}\n".format(old_query, response)
        prompt += "问：{}\n答：".format(query)
    # ---------------
    # tokenizer, this is an example of huggingface tokenizer.
    # input str, output['input_ids'] = tensor([[tokenized str, gmask, sop]])
    prompt, image_position, torch_image = process_image(prompt, image=image)
    if torch_image is not None:
        torch_image = torch_image.to(next(model.parameters()).dtype).to(next(model.parameters()).device)
    if image_position < 5: # no image
        inputs = tokenizer([prompt], return_tensors="pt").to(model.parameters().__next__().device)['input_ids'][0]
        pre_image = 0
    else:
        input0 = tokenizer.encode(prompt[:image_position], add_special_tokens=False)
        input1 = [tokenizer.pad_token_id] * model.image_length
        input2 = tokenizer.encode(prompt[image_position:], add_special_tokens=False)
        inputs = sum([input0, input1, input2], [])
        inputs = torch.tensor(tokenizer.build_inputs_with_special_tokens(inputs)).to(model.parameters().__next__().device)
        pre_image = len(input0)
    # ---------------
    # Next, we manually set the format to keep flexibility.
    mask_position = len(inputs) - 2
    context_length = len(inputs) - 1 # all before sop
    get_func = partial(get_masks_and_position_ids_glm, mask_position=mask_position, context_length=context_length)
    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    # ---------------
    # from sat.generation.sampling_strategies import BeamSearchStrategy
    # strategy = BeamSearchStrategy(num_beams, length_penalty=1., prefer_min_length=5, end_tokens=[tokenizer.eos_token_id], consider_end=True, no_repeat_ngram_size=5, stop_n_iter_unchanged=30, temperature=temperature, top_p=top_p, top_k=60, repetition_penalty=1.1)
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id],
                            invalid_slices=invalid_slices, repetition_penalty=repetition_penalty)
    output = filling_sequence(
        model, seq,
        batch_size=1,
        get_masks_and_position_ids=get_func,
        strategy=strategy,
        pre_image=pre_image,
        image=torch_image,
    )[0] # drop memory
    
    # ---------------
    # port from inference_glm.py, more general than chat mode
    # clip -1s and fill back generated things into seq
    if type(output) is not list:
        output_list = output.tolist()
    else:
        output_list = output
    for i in range(len(output_list)):
        output = output_list[i]
        if type(output) is not list:
            output = output.tolist()
        try:
            unfinished = output.index(-1)
        except ValueError:
            unfinished = len(output)
        if output[unfinished - 1] == tokenizer.eos_token_id:
            unfinished -= 1
        bog = output.index(tokenizer.bos_token_id)
        output_list[i] = output[:mask_position] + output[bog + 1:unfinished] + output[mask_position + 1:bog]
    # ---------------

    response = tokenizer.decode(output_list[0])
    sep = 'A:' if english else '答：'
    response = process_response(response).split(sep)[-1].strip()
    history = history + [(query, response)]
    return response, history, torch_image
