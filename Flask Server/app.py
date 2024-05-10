from flask import Flask, request
from flask_cors import CORS, cross_origin
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
import pymongo
from bson import ObjectId

import os

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

@app.route('/api/query', methods=['POST'])
def chat():
    data = request.get_json()
    answer = Respond(data['prompt'])
    newChat = {
        "question" : data['prompt'],
        "answer" : answer,
        "pTime" : data['tm']
    }
    session = sessionDoc.find(ObjectId(data['activeSessionID']))
    print(session)
    return answer

@app.route('/api/new-session', methods=['POST'])
def newSession():
    data = request.get_json()
    newDoc = {
        "user": data["token"],
        "name": "",
        "chats": [],
    }
    newS = sessionDoc.insert_one(newDoc)
    ans = str(newS.inserted_id)
    return ans

if __name__ == "__main__":
    app.run(debug=True, port=8080)
