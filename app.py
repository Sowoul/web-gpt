from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import requests
from bs4 import BeautifulSoup
from googlesearch import search

tokenizer = AutoTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")

chat_history = []


def gen(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors='pt')
    answer_start_scores, answer_end_scores = model(**inputs, return_dict=False)

    # Finding the tokens with the highest `start` and `end` scores
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Convert tokens to string
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0][answer_start:answer_end]))

    return answer


def get_urls(query, num_results=5):
    # Limit to the first num_results (default is 5)
    urls = [url for url in search(query, num_results == 5)]
    return urls


def generate_ctx(urls):
    ctx = ''
    for url in urls:

        r = requests.get(url)

        soup = BeautifulSoup(r.text, 'html.parser')

        p_tags = soup.find_all('p')

        for tag in p_tags:
            ctx = ctx+tag.get_text()
        return ctx[0:511]


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def fun():
    if request.method == 'POST':
        user_input = request.form['user_input']
        chat_history.append('User: ' + user_input)
        urs = get_urls(user_input)
        context = generate_ctx(urs)
        ai_response = gen(user_input, context=context)
        chat_history.append('AI: ' + ai_response)
    return render_template('index.html', chat_history=chat_history)


if __name__ == "__main__":
    app.run(port=8080)
