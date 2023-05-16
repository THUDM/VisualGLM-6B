#!/usr/bin/env python

import gradio as gr
from PIL import Image
import os
import json
from model import is_chinese, get_para, get_infer_setting, generate_input, get_para, chat

gpu_number = 0
model, tokenizer = get_infer_setting(gpu_device=gpu_number)

def generate_text_with_image(input_text, image, history=[], request_data=dict()):
    input_para = {
        "max_length": 512,
        "min_length": 50,
        "num_beams": 5,
        "temperature": 0.95,
        "top_p": 0.7,
        "repetition_penalty": 2.0,
        "length_penalty": 1.0,
        "early_stopping": True
    }
    input_para.update(request_data)

    input_data = generate_input(input_text, image, history, input_para, image_is_encoded=False)
    input_image, gen_kwargs =  input_data['input_image'], input_data['gen_kwargs']
    max_length, num_beams, top_p, temperature = get_para(gen_kwargs)
    answer, history = chat(None, model, tokenizer, input_text, history=history, image=input_image, \
                                     max_length=max_length, num_beams=num_beams, top_p=top_p, temperature=temperature)
        
    return answer


def request_model(input_text, temperature, top_p, image_prompt, result_previous):
    result_text = [(ele[0], ele[1]) for ele in result_previous]
    for i in range(len(result_text)-1, -1, -1):
        if result_text[i][0] == "" or result_text[i][1] == "":
            del result_text[i]
    print(f"history {result_text}")

    is_zh = is_chinese(input_text)
    if image_prompt is None:
        if is_zh:
            result_text.append((input_text, '图片为空！请上传图片并重试。'))
        else:
            result_text.append((input_text, 'Image empty! Please upload a image and retry.'))
        return input_text, result_text
    elif input_text == "":
        result_text.append((input_text, 'Text empty! Please enter text and retry.'))
        return "", result_text                

    request_para = {"temperature": temperature, "top_p": top_p}
    image = Image.open(image_prompt)
    try:
        answer = generate_text_with_image(input_text, image, result_text.copy(), request_para)
    except Exception as e:
        print(f"error: {e}")
        if is_zh:
            result_text.append((input_text, '超时！请稍等几分钟再重试。'))
        else:
            result_text.append((input_text, 'Timeout! Please wait a few minutes and retry.'))
        return "", result_text

    result_text.append((input_text, answer))
    print(result_text)
    return "", result_text


DESCRIPTION = '''# <a href="https://github.com/THUDM/VisualGLM">VisualGLM</a>'''

MAINTENANCE_NOTICE='Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.\nHint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'

NOTES = 'This app is adapted from <a href="https://github.com/THUDM/VisualGLM">https://github.com/THUDM/VisualGLM</a>. It would be recommended to check out the repo if you want to see the detail of our model and training process.'


def clear_fn(value):
    return "", [("", "Hi, What do you want to know about this image?")], None

def clear_fn2(value):
    return [("", "Hi, What do you want to know about this image?")]


def main():
    gr.close_all()
    examples = []
    with open("./examples/example_inputs.jsonl") as f:
        for line in f:
            data = json.loads(line)
            examples.append(data)


    with gr.Blocks() as demo:
        
        gr.Markdown(DESCRIPTION)
        gr.Markdown(MAINTENANCE_NOTICE)


        with gr.Row():
            with gr.Column():
                with gr.Group():
                    input_text = gr.Textbox(label='Input Text', placeholder='Please enter text prompt below and press ENTER.')
                    with gr.Row():
                        run_button = gr.Button('Generate')
                        clear_button = gr.Button('Clear')

                    image_prompt = gr.Image(type="filepath", label="Image Prompt", value=None)
                with gr.Row():
                    temperature = gr.Slider(maximum=1, value=0.95, minimum=0, label='Temperature')
                    top_p = gr.Slider(maximum=1, value=0.7, minimum=0, label='Top P')

            result_text = gr.components.Chatbot(label='Multi-round conversation History', value=[("", "Hi, What do you want to know about this image?")])


        gr_examples = gr.Examples(examples=[[example["text"], example["image"]] for example in examples], 
                                  inputs=[input_text, image_prompt],
                                  label="Example Inputs (Click to insert an examplet into the input box)",
                                  examples_per_page=3)

        gr.Markdown(NOTES)

        print(gr.__version__)
        run_button.click(fn=request_model,inputs=[input_text, temperature, top_p, image_prompt, result_text],
                         outputs=[input_text, result_text])
        input_text.submit(fn=request_model,inputs=[input_text, temperature, top_p, image_prompt, result_text],
                         outputs=[input_text, result_text])
        clear_button.click(fn=clear_fn, inputs=clear_button, outputs=[input_text, result_text, image_prompt])
        image_prompt.change(fn=clear_fn2, inputs=clear_button, outputs=[result_text])

        print(gr.__version__)

    demo.launch()


if __name__ == '__main__':
    main()