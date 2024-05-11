from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import pymongo
from bson.objectid import ObjectId
from bson import json_util

import os, json

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
    newChat = {
        "question" : data['prompt'],
        "answer" : answer,
        "pTime" : data['tm']
    }
    session = sessionDoc.find_one_and_update({ "_id" :ObjectId(data['sessionID'])}, {'$push': {'chats': newChat}})
    print("_______________")
    print(session)
    print("_______________")
    return answer

@app.route('/api/new-session', methods=['POST'])
def newSession():
    data = request.get_json()
    newDoc = {
        "user_id": data["user_id"],
        "cTime": data["timenow"],
        "chatHeading": "New Chat",
        "chatContent" : "Something brewing bois!!",
        "chats": [],
    }
    newS = sessionDoc.insert_one(newDoc)
    ans = str(ObjectId(newS.inserted_id))
    return ans

@app.route('/api/chatHistory/<user_id>', methods=['GET'])
def getData(user_id):
    print(user_id)
    retrieve = []
    for sessions in sessionDoc.find({"user_id": user_id}):
        retrieve.append(parse_json(sessions))
    topass = jsonify(retrieve)
    return topass

if __name__ == "__main__":
    app.run(debug=True, port=8080)
