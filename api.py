import os
import json
import uvicorn
from fastapi import FastAPI, Request
from model import get_para, get_infer_setting, generate_input, get_para, chat
import datetime

gpu_number = 0
model, tokenizer = get_infer_setting(gpu_device=gpu_number)


app = FastAPI()
@app.post('/v1/visual_glm')
async def visual_glm(request: Request):
    json_post_raw = await request.json()
    print("Start to process request")

    json_post = json.dumps(json_post_raw)
    request_data = json.loads(json_post)
    input_text, input_image_encoded, history = request_data['text'], request_data['image_prompt'], request_data['history']
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

    input_data = generate_input(input_text, input_image_encoded, history, input_para)
    input_image, gen_kwargs =  input_data['input_image'], input_data['gen_kwargs']
    max_length, num_beams, top_p, temperature = get_para(gen_kwargs)
    answer, history = chat(None, model, tokenizer, input_text, history=history, image=input_image, \
                                     max_length=max_length, num_beams=num_beams, top_p=top_p, temperature=temperature)
    
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "result": answer,
        "history": history,
        "status": 200,
        "time": time
    }
    return answer


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)