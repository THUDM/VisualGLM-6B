import torch
from sat.model.official import ChatGLMModel
from sat.model.finetune import PTuningV2Mixin
from sat.model.base_model import BaseMixin
from copy import deepcopy
import random
import json
from .blip2 import BLIP2

from sat.resources.urls import MODEL_URLS

MODEL_URLS['visualglm-6b'] = 'https://cloud.tsinghua.edu.cn/f/348b98dffcc940b6a09d/?dl=1'
MODEL_URLS['visualglm-6b-v1'] = 'https://cloud.tsinghua.edu.cn/f/f00d1ca2a4104803b0bf/?dl=1'


class ImageMixin(BaseMixin):
    def __init__(self, args):
        super().__init__()
        self.args = deepcopy(args)
        self.model = BLIP2(args.eva_args, args.qformer_args)

    def word_embedding_forward(self, input_ids, output_cross_layer, **kw_args):
        if kw_args["pre_image"] > input_ids.shape[1] or kw_args.get("image", None) is None:
            return self.transformer.word_embeddings(input_ids)
        image_emb = self.model(**kw_args) # TODO COMMENT IMAGE

        pre_id, pads, post_id = torch.tensor_split(input_ids, [kw_args["pre_image"], kw_args["pre_image"]+self.args.image_length], dim=1) # image after [Round 0]\n问：<img>
        pre_txt_emb = self.transformer.word_embeddings(pre_id)
        post_txt_emb = self.transformer.word_embeddings(post_id)
        return torch.cat([pre_txt_emb, image_emb, post_txt_emb], dim=1)

class VisualGLMModel(ChatGLMModel):
    def __init__(self, args, transformer=None, **kwargs):
        super().__init__(args, transformer=transformer, **kwargs)
        assert hasattr(args, 'image_length')
        self.image_length = args.image_length
        self.add_mixin("eva", ImageMixin(args))
        
        # self.del_mixin("chatglm-attn")
        # self.add_mixin("chatglm-attn", ChatGLMLoRAAttnMixin(args.hidden_size, args.num_attention_heads, args.num_layers, r=10))

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('VisualGLM', 'VisualGLM Configurations')
        group.add_argument('--image_length', type=int, default=32)
        group.add_argument('--eva_args', type=json.loads, default={})
        group.add_argument('--qformer_args', type=json.loads, default={})
        return super().add_model_specific_args(parser)
    
