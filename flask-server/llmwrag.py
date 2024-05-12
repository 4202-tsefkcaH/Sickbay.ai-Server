import os
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np
import time
import json
import warnings

import torch
from datasets import load_dataset
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from tqdm.notebook import tqdm

from trl import SFTTrainer
from huggingface_hub import interpreter_login
from langchain.docstore.document import Document as LangchainDocument
import pandas as pd
from typing import Optional, List, Tuple
from datasets import Dataset
from transformers import pipeline
from transformers import BitsAndBytesConfig
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from ragatouille import RAGPretrainedModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from transformers import Pipeline
import requests

os.environ["WANDB_DISABLED"] = "true"
os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"
warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {"Authorization": "Bearer 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"}


MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

EMBEDDING_MODEL_NAME="NeuML/pubmedbert-base-embeddings"

def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=False,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)


class llm:
    def __init__(self, docs=None):
        self.RAW_KNOWLEDGE_BASE = [
            LangchainDocument(page_content=doc["text"]) for doc in docs
        ]

        self.docs_processed = split_documents(
            100, 
            self.RAW_KNOWLEDGE_BASE,
            tokenizer_name=EMBEDDING_MODEL_NAME,
        ) 

        for i in range(len(self.docs_processed)):
            self.docs_processed[i].page_content = self.docs_processed[i].page_content.replace("\n"," ")

        self.KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
            self.docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )  

    def doc_data(self):
        return self.KNOWLEDGE_VECTOR_DATABASE
    
    def query(self,payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()     

    def answer_with_rag(
            self,
            question: str,
            knowledge_index: FAISS,
            reranker: Optional[RAGPretrainedModel] = None,
            num_retrieved_docs: int = 30,
            num_docs_final: int = 3,
        ) -> Tuple[str, List[LangchainDocument]]:
            # Gather documents with retriever
            print("=> Retrieving documents...")
            relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
            relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

            # Optionally rerank results
            if reranker:
                print("=> Reranking documents...")
                relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
                relevant_docs = [doc["content"] for doc in relevant_docs]

            relevant_docs = relevant_docs[:num_docs_final]

            # Build the final prompt
            context = "\nExtracted documents:\n"
            context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

            #final_prompt = RAG_PROMPT_TEMPLATE.format(question=question,context=context)

            # Redact an answer
            print("=> Generating answer...")
            qn=context
            s1= f"I will be giving some information regarding the patient's report based on the information given above and the knowledge you have, please answer the following query, even though it is not neccesary to use the information given above Context:- {qn} Query:-{question} Answer:- "
            output = self.query({
                "inputs": f"I will be giving some information regarding the patient's report based on the information given above and the knowledge you have, please answer the following query, even though it is not neccesary to use the information given above Context:- {qn} Query:-{question} Answer:- ",
            })
            t1=len(s1)
            answer = output[0]['generated_text'][t1:]
            return answer, relevant_docs

    def respond(self,question):
        start_time = time.time()
        answer, relevant_docs = self.answer_with_rag(question, self.KNOWLEDGE_VECTOR_DATABASE, reranker=None)
        print(time.time()-start_time)
        return answer, relevant_docs
    

# inst=llm(docs=[{'text': "Sample Written History and Physical Examination\n\nHistory and Physical Examination\nPatient Name: Rogers, Pamela\nReferral Source: Emergency Department\nChief Complaint & ID: Ms. Rogers is a 56 ylo WF\nhaving chest pains for the last week.\nHistory of Present Illness\n\nComments\n\nDate:\nData Source:\n\n6/2/04\nPatient\n\nDefine the reason for the patient's visit as who has been\n\nspecifically as possible.\n\nThis is the first admission for this 56 year old woman,\nwho states she was in her usual state of good health until\none week prior to admission. At that time she noticed the\nabrupt onset (over a few seconds to a minute) of chest pain\nwhich she describes as dull and aching in character. The\npain began in the left para-sternal area and radiated up to\nher neck. The first episode of pain one week ago occurred\nwhen she was working in her garden in the middle of the\nday. She states she had been working for approximately 45\nminutes and began to feel tired before the onset of the pain.\nHer discomfort was accompanied by shortness of breath, but\nno sweating, nausea, or vomiting. The pain lasted\napproximately 5 to 10 minutes and resolved when she went\ninside and rested in a cool area.\nSince that initial pain one week ago she has had 2 additional\nepisodes of pain, similar in quality and location to the first\nepisode. Three days ago she had a 15 minute episode of\npain while walking her dog, which resolved with rest. This\nevening she had an episode of pain awaken her from sleep,\nlasting 30 minutes, which prompted her visit to the\nEmergency Department. At no time has she attempted any\nspecific measures to relieve her pain, other than rest. She\ndescribes no other associated symptoms during these\nepisodes of pain, including dizziness, or palpitations. She\nbecomes short of breath during these\nepisodes but describes no other exertional dyspnea,\northopnea, or paroxysmal nocturnal dyspnea. No change in the pain\nwith movement, no association with food, no GERD SX, no palpable pain.\nShe has never been told she has heart problems, never had any\nchest pains before, does not have claudication. She was diagnosed with\nShe does not smoke nor does she have diabetes.\nShe was diagnosed with hypertension 3 years ago and had a\nTAH with BSO 6 years ago. She is not on hormone replacement\ntherapy. There is a family history of premature CAD.\nShe does not know her cholesterol level.\nPast Medical History\nSurgical -\n1994:\n1998:\n\nConvey the acute or chronic nature of the problem and\n\nestablish a chronology.\nonset\ncharacter\nlocation\nradiation\n\ncircumstances; exacerbating factors\nassociated symptoms\nduration\nresolution; alleviating factors\nDescribe the natural history of her problem since its\nonset\nChange or new circumstances to the problem\nNew duration\nReason she come in for visit\nWhat has patient tried for relief\n\nRelevant positive and negative ROS for this complaint\nReview of systems for the relevant organ system\nRelevant risk factorienvironmental conditions\n\nHTN 3 years ago,\n\nTotal abdominal hysterectomy and bilateral\noophorectomy for uterine fibroids.\nBunionectomy\n\nThis highly relevant, although it may seem like a\n\ntrivial detail at first\n\n",
#   'page_no': 0},
#  {'text': 'Medical History\n1998:\n1990:\n\nDiagnosed with hypertension and began on\nunknown medication. Stopped after 6 months\nbecause of drowsiness.\nDiagnosed with peptic ulcer disease, which\nresolved after three months on cimetidine. She\ndescribes no history of cancer, lung disease\nor previous heart disease.\n\nAlways use generic names\n\nAllergy: Penicillin; experienced rash and hives in 1985.\n\nAlways list the type of reported reaction\n\nSocial History -\nAlcohol use:\nTobacco use:\nMedications:\nFamily History\nMother:\nFather:\n\n1 or 2 beers each weekend; 1 glass of\nwine once a week with dinner.\nNone.\nNo prescription or illegal drug use.\nOccasional OTC ibuprofen (Advil) for\nheadache (QOD).\n79, alive and well.\n\nQuantity\n\nInclude over-the-counter drugs\n\nComment specifically on the presence or absence of\ndiseases relevant to the chief complaint\n\n54, deceased, heart attack. No brothers\nor sisters. There is a positive family history of\nhypertension, but no diabetes, or cancer.\n\nReview of Systems\nHEENT:\nproblems, or sore throat.\nCadiovascular:\nSee HPI\nGastrointestinal:\n\nNo complaints of headache change in vision, nose or ear\n\nSeparate each ROS section for easy identification\nOK to refer to HPI if adequately covered there\nList positive and negative findings in brief, concise\n\nNo complaints of dysphagia, nausea, vomiting, or change in\nstool pattern, consistency, or color. She complains of\nepigastric pain, burning in quality, approximately twice a\nmonth, which she notices primarily at night.\nNo complaints of dysuria, nocturia, polyuria, hematuria, or\nShe complains of lower back pain, aching in quality,\napproximately once every week after working in her garden.\nThis pain is usually relieved with Tylenol. She complains of\nno other arthralgias, muscle aches, or pains.\nShe complains of no weakness, numbness, or incoordination.\n\nphrases or sentences\n\nGenitourinary:\nvaginal bleeding.\nMusculoskeletal:\n\nNeurological:\n\n',
#   'page_no': 1},
#  {'text': 'Physical Examination\nVital Signs:\nTemperature 37 degrees.\nGeneral:\nSkin:\nHEENT:\nScalp normal.\n\nBlood Pressure 168/98, Pulse 90, Respirations 20,\nMs. Rogers appears alert, oriented and cooperative.\nNormal in appearance, texture, and temperature\n\nAlways list vital signs. Check for orthostatic BP/P\nchanges if it is relevant to the patient\'s complaint\nDescription may give very important clues as to the\nnature or severity of the patient\'s problem\nComment on all organ systems\nList specific normal or pathological findings when\nrelevant to the patient\'s complaint\n\nPupils equally round, 4 mm, reactive to light and\naccommodation, sclera and conjunctiva normal.\nFundoscopic examination reveals normal vessels without\nTympanic membranes and external auditory canals normal.\nOral pharynx is normal without erythema or exudate. Tongue\nEasily moveable without resistance, no abnormal adenopathy\nin the cervical or supraclavicular areas. Trachea is midline\nand thyroid gland is normal without masses. Carotid artery\nupstroke is normal bilaterally without bruits. Jugular venous\npressure is measured as 8 cm with patient at 45 degrees.\nLungs are clear to auscultation and percussion bilaterally\nexcept for crackles heard in the lung bases bilaterally. PMI\nis in the 5th inter-costal space at the mid clavicular line. A\ngrade 2/6 systolic decrescendo murmur is heard best at the\nsecond right inter-costal space which radiates to the neck.\nA third heard sound is heard at the apex. No fourth heart\nsound or rub are heard. Cystic changes are noted in the\nbreasts bilaterally but no masses or nipple discharge is\nThe abdomen is symmetrical without distention; bowel\nsounds are normal in quality and intensity in all areas; a\nbruit is heard in the right paraumbilical area. No masses or\nsplenomegaly are noted; liver span is 8 cm by percussion.\nNo cyanosis, clubbing, or edema are noted. Peripheral\npulses in the femoral, popliteal, anterior tibial, dorsalis pedis,\nbrachial, and radial areas are normal.\nNo palpable nodes in the cervical, supraclavicular, axillary\nNormal rectal sphincter tone; no rectal masses or\ntenderness. Stool is brown and guaiac negative. Pelvic\n\nhemorrhage.\n\nNasal mucosa normal.\nand gums are normal.\nNeck:\n\nChest:\n\nThis patient needs a detailed cardiac examination\n\nSeen.\nAbdomen:\n\nMore precise than saying "no hepatomegaly\n\nExtremities:\n\nNodes:\nor inguinal areas.\nGenital/Rectal:\n\nAlways include these exams, or comment specifically\n\nwhy they were omitted\n\n',
#   'page_no': 2},
#  {'text': 'exmaination reveals normal external genitalia, and normal\nvagina and cervix on speculum examination. Bimanual\nexamination reveals no palpable uterus, ovaries, or masses.\nCranial nerves II-XII are normal. Motor and sensory\nexamination of the upper and lower extremities is normal.\nGait and cerebellar function are also normal. Reflexes are\nnormal and symmetrical bilaterally in both extremities.\n\nNeurological:\n\nInitial Problem List\n1.\n2.\n3.\n4.\n5.\n6.\n7.\n8.\n9.\n10.\n11.\n12.\n13.\nRevised Problem List\n1.\n2.\n3.\n4.\n5.\n6.\n7.\n8.\n9.\n10.\n11.\n12.\n13.\n\nChest Pain\nDyspnea\nHistory of HTN (4 years)\nHistory of TAH/BSO\nHistory of peptic ulcer disease\nPenicillin allergy\nFH of early ASCVD\nEpigastric pain\nLow back pain\nHypertension\nSystolic murmur\nCystic changes of breasts\nAbdominal bruit\nChest pain\nFH of early ASCVD\nEarly surgical menopause\nDyspnea\nRecent onset HTN\nAbdominal bruit\nSystolic ejection murmur\nEpigastric pain\nHistory of peptic ulcer disease\nLumbosacral back pain\nOTC non-steroidal analgesic use\nCystic changes of breasts\nPenicillin allergy\n\nAlthough you can omit this initial problem list from your\nfinal written H&P, (and just list a final problem list\nshown below), it is useful to make an initial list simply\nto keep track of all problems uncovered in the interview\n(#1-9 in this list) and exam (#10-13)\n\nThis list regroups related problems (or those you\nsuspect are related) into a more logical sequence\n\n',
#   'page_no': 3},
#  {'text': "Assessment and Differential Diagnosis\n\n1.\n\nChest pain with features of angina pectoris\nThis patient's description of dull, aching, exertion\n\nrelated substernal chest pain is suggestive of ischemic cardiac\norigin. Her findings of a FH of early ASCVD, hypertension, and\nof coronary artery disease. Therefore, the combination of this\npatient's presentation, and the multiple risk factors make angina\npectoris the most likely diagnosis. The pain symptoms appear to\nbe increasing, and the occurrence of pain at rest suggests this\nfits the presentation of unstable angina, and hospitalization is\nare less likely. Gastro-esophageal reflux disease (GERD) may\noccur at night with recumbency, but usually is not associated with\nexertion. The pain of GERD is usually burning, and the patient\ndescribes no associated gastrointestinal symptoms such as\nnausea, vomiting or abdominal pain which might suggest peptic\nulcer disease. The presence of dyspnea might suggest a\npulmonary component to this patient's disease process, but\nthe absence of fever, cough or abnormal pulmonary examination\nfindings make a pulmonary infection less likely, and the\nassociation of the dyspnea with the chest pain supports the\ntheory that both symptoms may be from ischemic heart disease.\n\nIn the assessment you take each of the patient's major\npain is more likely due to ischemic heart disease instead\nof other possibilities. You tie in several of the other\nproblems as risk factors for ischemic heart disease, and\nnot merely as random unrelated problems. You should list\nand extensive justification for your most likely diagnosis.\nYou should also explain why you are less suspicious of\nalternative diagnoses, such as esophageal reflux disease,\npulmonary or musculoskeletal pain.\n\nearly surgical menopause are pertinent risk factors for development problems and draw conclusions, in this case that the chest\n\nindicated.\n\nOther processes may explain her chest pain, but\n\n2.\n\nDyspnea\n\nHer dyspnea may correlate with findings on physical\n\nexam of a third heart sound and pulmonary crackles,\nsuggesting left ventricular dysfunction. In that case, she may\nbe manifesting symptoms and findings of congestive heart\nfailure from myocardial ischemia.\n\nAs in the previous problem, you should explain, in more\ndetail than is shown in this example, why you felt the dyspnea\nis more likely to be from ischemic heart disease, and not\nasthma, bronchitis, or other possibilities. Follow this pattern\n\n3.\n\nRecent onset hypertension and abdominal bruit\nThis combination raises the possibility of a\n\nfor all subsequent problems.\n\nsecondary cause of hypertension, specifically ASCVD of the\nrenal artery leading to renovascular hypertension. The lack\nof hypertensive retinopathy and left ventricular hypertrophy\non physical examination further support a recent onset of her\n\nBP elevation.\n4.\n\nSystolic murmur\n\nThe possibility of important valvular heart disease\n\nis raised by the murmur, specifically, aortic stenosis. The\nmurmur radiates to the neck as an aortic valvular murmur\noften does, but a normal carotid upstroke may mean this\n\nmurmur is not significant.\n\n5.\n6.\n7.\n8.\n\nEpigastric discomfort, NSAID use with a history\n\nof peptic ulcer disease.\nLumbo-sacral back pain\nFibrocystic breast disease\nPenicillin allergy\n\n",
#   'page_no': 4},
#  {'text': "Plan:\n\n1. Carefully monitor the patient for any increased chest pain that\nmight be indicative of impending myocardial infarction by admitting\n\nYou should develop a diagnostic and therapeutic plan\nfor the patient. Your plan should incorporate acute and\nlong-term care of the patient's most likely problem. You\nshould consider pharmacologic and non-pharmacologic\nmuch as treating the disease when possible. You are\nexpected to know the usual classes of medications used\n\nthe patient to the telemetry floor.\n\n2. Start platelet inhibitors, such as aspirin to decrease the risk of measures and be cognizant of the fact that you need to\nmyocardial infarction; start nitrates to decrease the risk of occlusion treat the symptoms (i.e. make the patient comfortable) as\n\nand to treat her symptoms of pain. For prolonged pain un-\nresponsive to nitrates, she may need an analgesic such as\nmorphine. The nitrates will also help to lower her BP.\n3. Patient should have her cholesterol monitored and when\ndischarged she should be started on an appropriate exercise and\nweight loss program, including a low-fat diet. If her cholesterol\nis elevated, she may need cholesterol-lowering medication such\n4. Schedule a cardiac catheterization since non-invasive\ntests have a high pretest probability for being positive and regard-\nless of the result, negative or positive, she will need a cath\n5. Begin diuretics for her dyspnea which is most likely secondary\nto volume overload - this will treat her high BP as well. She should\nhave a ventriculogram with the cath that will assess cardiac size\nand presence of wall motion abnormalities.\n6. Appropriate lab work would include BUN/Creaunine\nto assess kidney function, electrolytes and baseline EKG.\n\nto treat these illnesses.\n\nas HMG Co-reductases.\n\n",
#   'page_no': 5}])

# ans, rd= inst.respond("What cure do you recommend?")
# print(ans)


"""
1. LLM bohot old h, 2018, other LLMs -> size, quantize kar payenge ki nahi, performance , finetuned on medical dataset
2. RAG integration -> chunking , stepwise prompting
Query:-

I will be giving some information regarding the patient's report in the next few prompts
Prompt1:- 2 chunks
Prompt2:- 3 chunks 
ROle System:- BAsed on the information given above and the knowledge you have, please answer the following query, even though it is not neccesary to use the information given above


3. Prompting techniques
"""