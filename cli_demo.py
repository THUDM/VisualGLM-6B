# -*- encoding: utf-8 -*-

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize

from model import VisualGLMModel, chat
from finetune_visualglm import FineTuneVisualGLMModel
from sat.model import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=100, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.8, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="visualglm-6b", help='pretrained ckpt')
    parser.add_argument("--prompt_zh", type=str, default="描述这张图片。", help='Chinese prompt for the first round')
    parser.add_argument("--prompt_en", type=str, default="Describe the image.", help='English prompt for the first round')
    args = parser.parse_args()

    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cuda' if (torch.cuda.is_available() and args.quant is None) else 'cpu',
    ))
    model = model.eval()

    if args.quant:
        quantize(model.transformer, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    if not args.english:
        print('欢迎使用 VisualGLM-6B 模型，输入图像URL或本地路径读图，继续输入内容对话，clear 重新开始，stop 终止程序')
    else:
        print('Welcome to VisualGLM-6B model. Enter an image URL or local file path to load an image. Continue inputting text to engage in a conversation. Type "clear" to start over, or "stop" to end the program.')
    with torch.no_grad():
        while True:
            history = None
            cache_image = None
            if not args.english:
                image_path = input("请输入图像路径或URL（回车进入纯文本对话）： ")
            else:
                image_path = input("Please enter the image path or URL (press Enter for plain text conversation): ")

            if image_path == 'stop':
                break
            if len(image_path) > 0:
                query = args.prompt_en if args.english else args.prompt_zh
            else:
                if not args.english:
                    query = input("用户：")
                else:
                    query = input("User: ")
            while True:
                if query == "clear":
                    break
                if query == "stop":
                    sys.exit(0)
                try:
                    response, history, cache_image = chat(
                        image_path, 
                        model, 
                        tokenizer,
                        query, 
                        history=history, 
                        image=cache_image, 
                        max_length=args.max_length, 
                        top_p=args.top_p, 
                        temperature=args.temperature,
                        top_k=args.top_k,
                        english=args.english,
                        invalid_slices=[slice(63823, 130000)] if args.english else []
                        )
                except Exception as e:
                    print(e)
                    break
                sep = 'A:' if args.english else '答：'
                print("VisualGLM-6B："+response.split(sep)[-1].strip())
                image_path = None
                if not args.english:
                    query = input("用户：")
                else:
                    query = input("User: ")


if __name__ == "__main__":
    main()