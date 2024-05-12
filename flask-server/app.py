from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import pymongo
from bson.objectid import ObjectId
from bson import json_util

import os, json

from llmwrag import llm
from ocr import pdf2text

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

client = pymongo.MongoClient("mongodb+srv://csratul03:bw1vpLYjCm3X7BZ8@cluster0.hvrjuwn.mongodb.net/")
db = client['test']
sessionDoc = db['session-data']

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def parse_json(data):
    return json.loads(json_util.dumps(data))


@app.route('/api/query', methods=['POST'])
def chat():
    data = request.get_json()
    st = "Question:- " + data['prompt'] + "\n Answer:- "
    answer = query({"inputs": st})[0]['generated_text']
    return answer

@app.route('/api/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    pdf_file = request.files['file']
    print("request: ", pdf_file)
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        pdf_bytes = pdf_file.read()
        text = pdf2text(pdf_bytes)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/llm', methods=['POST'])
def llmAPI():
    data = request.get_json()
    inst = llm(docs=data['text'])
    ans, rd = inst.respond(data['prompt'])
    return ans

if __name__ == "__main__":
    app.run(debug=True, port=8080)
