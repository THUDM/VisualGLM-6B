"""
Fay数字人开源项目发起人 郭泽斌 于20230519日在广州补充。
因windows 下sat安装失败，故补充hf的加载方式。
"""

import os
import platform
import json
import signal
from transformers import AutoTokenizer, AutoModel
import uvicorn
from fastapi import FastAPI, Request

tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history, prefix):
    prompt = prefix
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

app = FastAPI()
@app.post('/')
async def visual_glm(request: Request):
    json_post_raw = await request.json()
    print("Start to process request")

    json_post = json.dumps(json_post_raw)
    request_data = json.loads(json_post)
    global stop_stream
 
    history = request_data.get("history")
    image_path = request_data.get("image")
    query = request_data.get("text")
    while True:
        count = 0
        for response, history in model.stream_chat(tokenizer, image_path, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            else:
                count += 1
                if count % 8 == 0:
                    os.system(clear_command)
                    print(build_prompt(history, ""), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history, ""), flush=True)
        return response
           

    
if __name__ == "__main__":
   uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)