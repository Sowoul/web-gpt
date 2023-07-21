from flask import Flask, render_template, request, jsonify
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
    # Split the context into chunks of size 510 (leaving 2 tokens for [CLS] and [SEP])
    context_parts = [context[i:i+510] for i in range(0, len(context), 510)]

    # Initialize lists to store the scores and answers
    answer_start_scores = []
    answer_end_scores = []
    answers = []

    # Process each chunk separately
    for context_part in context_parts:
        inputs = tokenizer.encode_plus(
            question, context_part, return_tensors='pt')
        start_scores, end_scores = model(**inputs, return_dict=False)

        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores) + 1

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end]))

        answer_start_scores.append(start_scores[0][answer_start].item())
        answer_end_scores.append(end_scores[0][answer_end-1].item())
        answers.append(answer)

    # Return the answer with the highest start score
    return answers[answer_start_scores.index(max(answer_start_scores))]


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
        return ctx


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
        return jsonify({'message': ai_response})
    return render_template('index.html', chat_history=chat_history)


if __name__ == "__main__":
    app.run(port=8080)

