from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
import pymongo
from bson.objectid import ObjectId
from bson import json_util

import os, json

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

client = pymongo.MongoClient("mongodb+srv://csratul03:bw1vpLYjCm3X7BZ8@cluster0.hvrjuwn.mongodb.net/")
db = client['test']
sessionDoc = db['session-data']

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
hub_llm = HuggingFaceHub(repo_id="openai-community/gpt2", model_kwargs={"max_length":256})

prompt = PromptTemplate(
    input_variables=["question"],
    template="Answer: {question}"
)

hub_chain=LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

def Respond(query:str="How are you? "):
    return hub_chain.run(query)

def parse_json(data):
    return json.loads(json_util.dumps(data))

@app.route('/api/query', methods=['POST'])
def chat():
    data = request.get_json()
    print(data)
    answer = Respond(data['prompt'])
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
