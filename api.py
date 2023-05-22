import os
import json
import uvicorn
from fastapi import FastAPI, Request
from model import is_chinese, get_infer_setting, generate_input, chat
import datetime
import torch

gpu_number = 0
model, tokenizer = get_infer_setting(gpu_device=gpu_number)

app = FastAPI()
@app.post('/')
async def visual_glm(request: Request):
    json_post_raw = await request.json()
    print("Start to process request")

    json_post = json.dumps(json_post_raw)
    request_data = json.loads(json_post)
    input_text, input_image_encoded, history = request_data['text'], request_data['image'], request_data['history']
    input_para = {
        "max_length": 2048,
        "min_length": 50,
        "temperature": 0.8,
        "top_p": 0.4,
        "top_k": 100,
        "repetition_penalty": 1.2
    }
    input_para.update(request_data)

    is_zh = is_chinese(input_text)
    input_data = generate_input(input_text, input_image_encoded, history, input_para)
    input_image, gen_kwargs =  input_data['input_image'], input_data['gen_kwargs']
    with torch.no_grad():
        answer, history, _ = chat(None, model, tokenizer, input_text, history=history, image=input_image, \
                            max_length=gen_kwargs['max_length'], top_p=gen_kwargs['top_p'], \
                            top_k = gen_kwargs['top_k'], temperature=gen_kwargs['temperature'], english=not is_zh)
        
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    response = {
        "result": answer,
        "history": history,
        "status": 200,
        "time": time
    }
    return response


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)