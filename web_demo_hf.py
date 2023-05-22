from transformers import AutoModel, AutoTokenizer
import gradio as gr
import mdtex2html
import torch

"""Override Chatbot.postprocess"""

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y

gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, image_path, chatbot, max_length, top_p, temperature, history):
    if image_path is None:
        return [(input, "图片不能为空。请重新上传图片并重试。")], []
    chatbot.append((parse_text(input), ""))
    with torch.no_grad():
        for response, history in model.stream_chat(tokenizer, image_path, input, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
            chatbot[-1] = (parse_text(input), parse_text(response))

            yield chatbot, history


def predict_new_image(image_path, chatbot, max_length, top_p, temperature):
    input, history = "描述这张图片。", []
    chatbot.append((parse_text(input), ""))
    with torch.no_grad():
        for response, history in model.stream_chat(tokenizer, image_path, input, history, max_length=max_length,
                                               top_p=top_p,
                                               temperature=temperature):
            chatbot[-1] = (parse_text(input), parse_text(response))

            yield chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return None, [], []


DESCRIPTION = '''<h1 align="center"><a href="https://github.com/THUDM/VisualGLM-6B">VisualGLM</a></h1>'''
MAINTENANCE_NOTICE = 'Hint 1: If the app report "Something went wrong, connection error out", please turn off your proxy and retry.\nHint 2: If you upload a large size of image like 10MB, it may take some time to upload and process. Please be patient and wait.'
NOTES = 'This app is adapted from <a href="https://github.com/THUDM/VisualGLM-6B">https://github.com/THUDM/VisualGLM-6B</a>. It would be recommended to check out the repo if you want to see the detail of our model and training process.'

def main(args):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
    if args.quant in [4, 8]:
        model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).quantize(args.quant).half().cuda()
    else:
        model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()
    model = model.eval()

    with gr.Blocks(css='style.css') as demo:
        gr.HTML(DESCRIPTION)
        
        with gr.Row():
            with gr.Column(scale=2):
                image_path = gr.Image(type="filepath", label="Image Prompt", value=None).style(height=504)
            with gr.Column(scale=4):
                chatbot = gr.Chatbot().style(height=480)
        with gr.Row():
            with gr.Column(scale=2, min_width=100):
                max_length = gr.Slider(0, 4096, value=2048, step=1.0, label="Maximum length", interactive=True)
                top_p = gr.Slider(0, 1, value=0.4, step=0.01, label="Top P", interactive=True)
                temperature = gr.Slider(0, 1, value=0.8, step=0.01, label="Temperature", interactive=True)
            with gr.Column(scale=4):
                with gr.Box():
                    with gr.Row():
                        with gr.Column(scale=2):
                            user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=4).style(
                                container=False)
                        with gr.Column(scale=1, min_width=64):
                            submitBtn = gr.Button("Submit", variant="primary")
                            emptyBtn = gr.Button("Clear History")
                    gr.Markdown(MAINTENANCE_NOTICE + '\n' + NOTES)
        history = gr.State([])
        

        submitBtn.click(predict, [user_input, image_path, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                        show_progress=True)
        image_path.upload(predict_new_image, [image_path, chatbot, max_length, top_p, temperature], [chatbot, history],
                        show_progress=True)
        image_path.clear(reset_state, outputs=[image_path, chatbot, history], show_progress=True)
        submitBtn.click(reset_user_input, [], [user_input])
        emptyBtn.click(reset_state, outputs=[image_path, chatbot, history], show_progress=True)

        print(gr.__version__)

        demo.queue().launch(share=args.share, inbrowser=True, server_name='0.0.0.0', server_port=8080)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    main(args)
