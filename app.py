from transformers import pipeline
import requests
from googlesearch import search
qa_pipeline = pipeline('question-answering',
                       model='distilbert-base-uncased-distilled-squad')


def search_context(question):
    search_results = search(question, num_results=5, lang='en', advanced=True)
    descriptions = [str(result.description.encode(
        'utf-8')) for result in search_results]

    context = " ".join(descriptions)

    return context


def answer_question(question):
    context = search_context(question)
    answer = qa_pipeline({
        'context': context,
        'question': question
    })

    return answer['answer']


question = "How do i add something to a python list"
print(answer_question(question))
