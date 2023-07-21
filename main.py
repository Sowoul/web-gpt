from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from flask import Flask, render_template, request
from transformers import GPT2Tokenizer, GPTNeoForQuestionAnswering, GenerationConfig
import torch

chat_history = []


def gen(tkn_inp: str):
    tkn_inp = tkn_inp.replace('%', ' ')
    print(tkn_inp)

    def encode_input(input_str):
        return tokenizer(input_str, return_tensors="pt")

    def get_generation_config(eos_token_id):
        generation_config = GenerationConfig(
            max_length=512,
            pad_token_id=eos_token_id,
            do_sample=True
        )
        return generation_config

    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model.eval()

    def generate_text(inputs):

        generation_config = get_generation_config(model.config.eos_token_id)

        with torch.no_grad():
            prediction = model.generate(
                **inputs, **generation_config.to_dict())

        return tokenizer.decode(prediction[0], skip_special_tokens=True)

    inputs = encode_input(tkn_inp)
    output = generate_text(inputs)
    return output


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def fun():
    if request.method == 'POST':
        user_input = request.form['user_input']
        chat_history.append('User: ' + user_input)
        ai_response = gen(user_input)
        chat_history.append('AI: ' + ai_response)
    return render_template('index.html', chat_history=chat_history)


if __name__ == "__main__":
    app.run(port=8080)
