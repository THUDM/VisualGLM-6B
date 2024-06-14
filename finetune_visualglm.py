import os
import torch
import argparse
from functools import partial
import json
from tqdm import tqdm

from sat import mpu, get_args, get_tokenizer
from sat.training.deepspeed_training import training_main
from model import VisualGLMModel
from sat.model.finetune import PTuningV2Mixin
from sat.model.finetune.lora2 import LoraMixin
from sat.generation.autoregressive_sampling import filling_sequence, BaseStrategy
from sat.model.mixins import CachedAutoregressiveMixin

from model import chat, VisualGLMModel

from torch.nn import CrossEntropyLoss
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

    
def forward_step_eval(data_iterator, model, args, timers):
    # Metric
    model.eval()
    
    try:
        model.add_mixin('auto-regressive', CachedAutoregressiveMixin())   
    except Exception as e:
        # Avoid assertion errors caused by duplicate additions
        pass
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = preds
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred))
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict
    
    # Get the batch.
    tokens, labels, image, pre_image = get_batch(data_iterator, args, timers)
    
    gen_kwargs = { "max_length": 1024, "top_p": 0.4, "top_k":100, "temperature": 0.8}

    with open(args.valid_data[0], 'r', encoding='utf-8') as file:
        data = json.load(file)
    outputs = []
    with torch.no_grad():
        for item in tqdm(data):
            response, _, _ = chat(
                            item['img'], 
                            model, 
                            tokenizer,
                            item['prompt'], 
                            history=[], 
                            max_length=gen_kwargs['max_length'], 
                            top_p=gen_kwargs['top_p'], 
                            temperature=gen_kwargs['temperature'],
                            top_k=gen_kwargs['top_k'],
                            )
            sep = '答：'
            outputs.append(response.split(sep)[-1].strip())
    return torch.tensor(0, device=torch.device('cuda')), {k:torch.tensor(v, device=torch.device('cuda')) for k,v in compute_metrics((outputs, labels.cpu())).items()}



class FineTuneVisualGLMModel(VisualGLMModel):
    def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kw_args)
        if args.use_ptuning:
            self.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.pre_seq_len))
        if args.use_lora:
            self.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range), reinit=True)
            # self.get_mixin("eva").model.glm_proj = replace_linear_with_lora(self.get_mixin("eva").model.glm_proj, LoraLinear, args.lora_rank)
        elif args.use_qlora:
            self.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, layer_range=args.layer_range, qlora=True), reinit=True)
        self.args = args
        
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('VisualGLM-finetune', 'VisualGLM finetune Configurations')
        group.add_argument('--pre_seq_len', type=int, default=8)
        group.add_argument('--lora_rank', type=int, default=10)
        group.add_argument('--use_ptuning', action="store_true")
        group.add_argument('--use_lora', action="store_true")
        group.add_argument('--use_qlora', action="store_true")
        group.add_argument('--layer_range', nargs='+', type=int, default=None)
        return super().add_model_specific_args(parser)

    def disable_untrainable_params(self):
        enable = []
        if self.args.use_ptuning:
            enable.extend(['ptuning'])
        if self.args.use_lora or self.args.use_qlora:
            enable.extend(['matrix_A', 'matrix_B'])

        
        for n, p in self.named_parameters():
            flag = False
            for e in enable:
                if e.lower() in n.lower():
                    flag = True
                    break
            if not flag:
                p.requires_grad_(False)
            else:
                print(n)


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'labels']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    data_i = mpu.broadcast_data(['image'], data, torch.float32)
    # Unpack.
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()
    img = data_i['image']
    if args.fp16:
        img = img.half()
    
    return tokens, labels, img, data['pre_image']


from torch.nn import CrossEntropyLoss

def forward_step(data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, image, pre_image = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    logits = model(input_ids=tokens, image=image, pre_image=pre_image)[0]
    dtype = logits.dtype
    lm_logits = logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    lm_logits = lm_logits.to(dtype)
    loss = loss.to(dtype)
    return loss, {'loss': loss}


from model.blip2 import BlipImageEvalProcessor
from torch.utils.data import Dataset
import json
from PIL import Image

class FewShotDataset(Dataset):
    def __init__(self, path, processor, tokenizer, args):
        max_seq_length = args.max_source_length + args.max_target_length
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.images = []
        self.input_ids = []
        self.labels = []
        for item in data:
            image = processor(Image.open(item['img']).convert('RGB'))
            input0 = tokenizer.encode("<img>", add_special_tokens=False)
            input1 = [tokenizer.pad_token_id] * args.image_length
            input2 = tokenizer.encode("</img>问："+item['prompt']+"\n答：", add_special_tokens=False)
            a_ids = sum([input0, input1, input2], [])
            b_ids = tokenizer.encode(text=item['label'], add_special_tokens=False)
            if len(a_ids) > args.max_source_length - 1:
                a_ids = a_ids[: args.max_source_length - 1]
            if len(b_ids) > args.max_target_length - 2:
                b_ids = b_ids[: args.max_target_length - 2]
            pre_image = len(input0)
            input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)

            context_length = input_ids.index(tokenizer.bos_token_id)
            mask_position = context_length - 1
            labels = [-100] * context_length + input_ids[mask_position+1:]
            
            pad_len = max_seq_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels = labels + [tokenizer.pad_token_id] * pad_len
            if args.ignore_pad_token_for_loss:
                labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
            self.images.append(image)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
        self.pre_image = pre_image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
            "pre_image": self.pre_image
        }


def create_dataset_function(path, args):
    tokenizer = get_tokenizer(args)
    image_processor = BlipImageEvalProcessor(224)

    dataset = FewShotDataset(path, image_processor, tokenizer, args)
    return dataset


if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--max_source_length', type=int)
    py_parser.add_argument('--max_target_length', type=int)
    py_parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True)
    # py_parser.add_argument('--old_checkpoint', action="store_true")
    py_parser.add_argument('--source_prefix', type=str, default="")
    py_parser = FineTuneVisualGLMModel.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.device = 'cpu'

    model_type = 'visualglm-6b'
    model, args = FineTuneVisualGLMModel.from_pretrained(model_type, args)
    if torch.cuda.is_available():
        model = model.to('cuda')
    tokenizer = get_tokenizer(args)
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    def data_collator(examples):
        for example in examples:
            example['input_ids'] = torch.tensor(example['input_ids'], dtype=torch.long)
            example['labels'] = torch.tensor(example['labels'], dtype=torch.long)
        ret = {
            'input_ids': torch.stack([example['input_ids'] for example in examples]),
            'labels': torch.stack([example['labels'] for example in examples]),
            'image': torch.stack([example['image'] for example in examples]),
            'pre_image': example['pre_image']
        }
        return ret
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function, collate_fn=data_collator, forward_step_eval=forward_step_eval)
