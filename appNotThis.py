

import os
import openai
import time
import numpy as np
import matplotlib.pyplot as plt
import re

from llm_challenge.utils.misc import set_openai_api_key,\
                                    misc_get_completion, \
                                    read_dict_from_json, \
                                    write_dict_to_json,\
                                    read_text 
                                                
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
# from llm_challenge.utils.misc import read_dict_from_json, \
#                                      set_openai_api_key, \
#                                      write_dict_to_json, \
#                                      get_completion, \
#                                      read_text
from langchain.callbacks import get_openai_callback


from appHelperFunction import get_completion,\
                                get_embedding, \
                                annotate_question,\
                                translate_response,\
                                reason_given_context


# _ = load_dotenv(find_dotenv())
# openai.api_key = os.environ['OPENAI_API_KEY']
set_openai_api_key()
#set the path to the challenge dataset
# DATA_DIR = '../data' #not working on my mac
# DATA_DIR  = '/Users/arnelcatangay/Documents/GitHub/llm_challenge_main' #not working on my mac
DATA_DIR = 'data'
# import sys
# sys.path.append("/Users/arnelcatangay/Documents/GitHub/llm_challenge_as23-main")
dset = 'tutorial'


    



qas_fname = f"{DATA_DIR}/qas/qas_{dset}.json"
qas_dict = read_dict_from_json(qas_fname)
qas_dict["47"]
gpt3p5_answers = {}
max_retries = 3
for q_id, qa_dict in (pbar := tqdm(qas_dict.items())):
    num_retries = 0
    while num_retries < max_retries:
        try:
            pbar.set_description(f"Answering question {q_id}")
            # gpt3p5_answers[q_id] = get_completion(qa_dict["question"])
            break
        except Exception as e:
            num_retries +=1
            gpt3p5_answers[q_id] = 'I do not know.'

write_dict_to_json(f"gpt3p5_{dset}.json", gpt3p5_answers)


q_id = "50"
qa_dict = qas_dict[q_id]
print(qa_dict)
# get_completion(qa_dict["question"])
contexts_fname = f"{DATA_DIR}/qas/contexts_{dset}.json"
contexts_dict = read_dict_from_json(contexts_fname)
context = contexts_dict[q_id]
question = qa_dict["question"]
prompt = """
I will give you a piece of text from electrical/electronic module's datashet delimted by ---
Your task is to give a detailed answer to the question that follows the text based on the provided text.
If the text does not contain sufficient information to answer, just say I do not know.

Here is the text:
---
{context}
---
Question: {question}
Answer:
"""

get_completion(prompt.format(context=context, question=question), is_chat_model=True)




qas_dict = read_dict_from_json(f"{DATA_DIR}/qas/qas_{dset}.json")

q_id = "50"
qas_dict[q_id]
# let's consider the pool of datasheets from which answers were taken...
doc_fnames = list(set([DATA_DIR + "/datasheets/" + qa_dict["datasheet"]  for qa_dict in qas_dict.values()]))
doc_fnames
# let's pull the datasheet for question 50
doc_idx = doc_fnames.index(str(DATA_DIR + "/datasheets/parsed/ltc7804.txt"))
print(doc_fnames[doc_idx])
text = read_text(doc_fnames[doc_idx])


prompt = """
I will give you a peice of text from electrical/electronic module's datashet delimted by ---
Your task is to give a detailed answer to the question that follows the text based on the provided text.
If the text does not contain sufficient information to answer, just output `IDK`.

Here is the text:
---
{text}
---
Question: {question}
Answer:
"""
# MAXIMUM TOKEN FAILURE!
#get_completion(prompt.format(text=text, question=qas_dict[q_id]["question"]))


# one token is roughly 4 characters (let's be over-conservative)
context_length = 2 * 4000
# get_completion(prompt.format(text=text[:context_length], question=qas_dict[q_id]["question"]))
start_idx = text.rfind('C1 value')
# get_completion(prompt.format(text=text[start_idx-context_length // 2: start_idx + context_length//2], question=qas_dict[q_id]["question"]))
qas_dict[q_id]["answer"]



### Embeddings 



texts = [
    "Boston is in the US.",
    "Paris is in France.",
    "Rome is in Italy.",
    "ahead of what is possible.",
    "it will rain tomorrow",
    "it rained a lot yesterday",
    "it is pouring"
]
embeddings = [get_embedding(text) for text in texts]
relatedness = [[se.dot(te) for te in embeddings] for se in embeddings]

plt.matshow(relatedness,cmap=plt.cm.Greens)
_ = plt.colorbar()
_ = plt.title("Similarity matrix of texts listed above")
embeddings_fname = f"{DATA_DIR}/qas/embeddings_{dset}.npz"
is_ef_exists = Path(embeddings_fname).exists()
print("Embeddings file does" + " exist." if is_ef_exists else " not exist.")
print("Set `is_generate_ef` to True if you would like to recompute it.")
is_generate_ef = False

if is_ef_exists and not is_generate_ef:
    print("loading embeddings from disk")
    # if we have previously created the emebeddings
    embeddings_data = np.load(embeddings_fname)
    embeddings, texts = embeddings_data["embeddings"], embeddings_data["texts"]
elif is_generate_ef:
    # let's read all texts and split them into chunks of permissible context length
    texts = []
    for df in doc_fnames:
        text = read_text(df)
        texts += [text[i:i+context_length] for i in range(0, len(text), context_length)]

    embeddings = []
    for text in texts:
        embeddings.append(get_embedding(text))

    # save
    np.savez(embeddings_fname, embeddings=embeddings, texts=texts)
else:
    print("""It looks like there no embeddings file.
    Set `is_generate_ef` to True to create it.""")
question = qas_dict[q_id]["question"]
question_embedding = get_embedding(question)

context_embeddings = np.array(embeddings)
# for idx, text in enumerate(texts):
#     if text.rfind('C1 value')> -1:
#         print(idx)
query_embedding = question_embedding
relatedness_score = context_embeddings.dot(query_embedding)
_ = plt.plot(relatedness_score)
top_idxs = np.argsort(-relatedness_score)
selected_by_relatedness_context = texts[top_idxs[5]]
relatedness_score[top_idxs]
# get_completion(prompt.format(text=texts[top_idxs[0]], question=question))
# get_completion(prompt.format(text=texts[top_idxs[5]], question=question))
# something like iterate over the top retrieved texts till I do not know is not returned.
for _ in range(10):
    answer = get_completion(prompt.format(text=texts[top_idxs[_]], question=question))
    answer = 'IDK'
    if 'IDK' not in answer: 
        break
    else:
        print("Retrieved text does not contain sufficient information.")
print(answer)


MODEL_COST_PER_1K_TOKENS_GPT3P5 = 0.0015
_, num_tokens = misc_get_completion(prompt.format(text=texts[top_idxs[0]], question=question), is_return_total_tokens=True)
print(num_tokens)
# num_tokens = 2869
TOP_K = 1000
CONTEXT_LENGTH_IN_K_TOKENS = num_tokens * 1e-3
NUM_QUESTIONS = 1000
# MODEL_COST_PER_1K_TOKENS_GPT3P5 * TOP_K * CONTEXT_LENGTH_IN_K_TOKENS * NUM_QUESTIONS
total_cost_fn = lambda num_tokens: 1e-3 * MODEL_COST_PER_1K_TOKENS_GPT3P5 *  num_tokens 

print(f"Total cost for LLM calls: ${total_cost_fn(num_tokens) * TOP_K * NUM_QUESTIONS}")

#TODO: migrate to helper function
def get_answer(question, top_k=6):
    question_embedding = get_embedding(question)
    relatedness_score = context_embeddings.dot(question_embedding)
    top_idxs = np.argsort(-relatedness_score)
    total_num_tokens = 0
    for _ in range(top_k):
        selected_by_relatedness_context = texts[top_idxs[_]]
        answer, num_tokens = misc_get_completion(prompt.format(
            text=selected_by_relatedness_context, 
            question=question),is_return_total_tokens=True)
        total_num_tokens += num_tokens
        if 'IDK' in answer:
            continue
        else:
            break
    return answer, total_num_tokens  
    
# let's try answer all question in the dataset:

answers_with_numpy = {}
max_retries = 5
total_num_tokens = 0

for q_id, qa_dict in tqdm(qas_dict.items()):
    num_tries = 0
    while num_tries < max_retries:
        try:
            time.sleep(0.5)
            answer, num_tokens = get_answer(qa_dict["question"])
            total_num_tokens += num_tokens
            break
        except Exception as e:
            answer = 'can not answer.'
            num_tries +=1

    answers_with_numpy[q_id] = answer
print(f"Finished answering questions using (USD) ${total_cost_fn(total_num_tokens):.5f} ")
write_dict_to_json(f"gpt3p5-with-top6-embeddings_{dset}.json", answers_with_numpy)
total_num_tokens




qas_dict = read_dict_from_json(f"{DATA_DIR}/qas/qas_{dset}.json")
doc_fnames = list(set([str(DATA_DIR + "/datasheets/" + datum["datasheet"])  for datum in qas_dict.values()]))
# Load and process the text
loaders = [TextLoader(_) for _ in doc_fnames]
documents = [d for loader in loaders for d in loader.load()]
# Next we split documents into small chunks. This is so we can find the most relevant chunks for a query and pass only those into the LLM.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
# Initialize PeristedChromaDB

# Create embeddings for each chunk and insert into the Chroma vector database. The persist_directory argument tells ChromaDB where to store the database when it's persisted.
# Supplying a persist_directory will store the embeddings on disk
persist_directory = f'{DATA_DIR}/qas/db_{dset}'
embedding = OpenAIEmbeddings()
# Embed and store the texts
# Persist the Database
# In a notebook, we should call persist() to ensure the embeddings are written to disk. 
# This isn't necessary in a script - the database will be automatically persisted when the client object is destroyed.

if Path(persist_directory).exists():
    print("Database exists... loading from disk")
    # Load the Database from disk, and create the chain
    # Be sure to pass the same persist_directory and embedding_function as 
    # you did when you instantiated the database. Initialize the chain we will use for question answering.
    # Now we can load the persisted database from disk, and use it as normal. 
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
else:
    print("Database does not exist. Do you want to recompute it?")
    print("If yes, set `is_generate_db` to True in the next cell.")
# to save from token-expensive ops (set to False)
is_generate_db = False
if is_generate_db:
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()
# create the retrieval QA chain with `stuff` mode
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever())
# Ask questions!

# Now we can use the chain to ask questions!




answers_with_retrieval = {}
with get_openai_callback() as cb:
    for q_id, qa_dict in (pbar := tqdm(qas_dict.items())):
        pbar.set_description(f"Answering question {q_id}")
        answer = qa.run(qa_dict["question"])
        answers_with_retrieval[q_id] = answer
    print(cb)

COST_PER_1K_TOKENS_DAV = 0.02
print(f"${6334 * COST_PER_1K_TOKENS_DAV * 1e-3:.5f}")




qas_dict = read_dict_from_json(f"{DATA_DIR}/qas/qas_{dset}.json")
doc_fnames = list(set([DATA_DIR + "/datasheets/" + datum["datasheet"]  for datum in qas_dict.values()]))
module_names = [_.split('/')[-1][:-4] for _ in doc_fnames]
#### REACT


instruction = """Solve a question answering task about datasheets with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be five types: 
(1) Search[`module_name`], which searches for the name of electrical/electronic component `module_name` mentioned in the question. If not, it will return UNK.
(2) Annotate, which lists no more than three keywords that distinctively describe the `question`.
(3) LookupKeywords, which returns a piece of text containing the list of keywords.
(4) Reason, which provides a response to the `question` based on the returned text.
(5) Finish, which returns the response (from reasoning step) and finishes the task if answer is known or not.
Here is one example.
"""

examples = """
Question: What is the purpose of the Conditional Search ROM command in the DS28E04-100 module?
Thought 1: I need to find the datasheet of the DS28E04-100 module.
Action 1: Search[DS28E04-100]
Observation 1: datasheet found.
Thought 2: Now I need to look up the paragraph that talks about Conditional Search ROM command. Let me provide some keywords to find this paragraph.
Action 2: Annotate
Observation 2: Conditional Search, ROM, command
Thought 3: Let me look up the paragraph that contains these keywords
Action 3: LookupKeywords
Observation 3: Relevant text found.
Thought 4: Now I need to reason about the question given the found text.
Action 4: Reason
Observation 4: The purpose of the Conditional Search ROM command in the DS28E04-100 module is to allow the bus master to identify devices on a multidrop system that fulfill certain conditions (CSR = 1) and have to signal an important event. Only those devices that fulfill the conditions will participate in the search. After each pass of the conditional search that successfully determined the 64-bit ROM code for a specific device on the multidrop bus, that particular device can be individually accessed as if a Match ROM had been issued, since all other devices will have dropped out of the search process and will be waiting for a reset pulse.
Thought 5: I think this answers the question.
Action 5: Finish

Start...

Question: {question}
"""

# def get_embedding(text, model="text-embedding-ada-002"):
#     text = text.replace("\n", " ")
#     return np.array(openai.Embedding.create(input = [text], model=model)['data'][0]['embedding'])
    
#TODO: migrate to helper function
def lookup_keywords(module_name, doc_idx, keywords):
    if doc_idx > -1:
        candidate_text = read_text(doc_fnames[doc_idx])
    else:
        candidate_texts = [text for text in texts \
                           if np.any([kw in text for kw in keywords])\
                          ]
        candidate_text = " ".join(candidate_texts)
    context_window = 5000
    candidate_texts = [candidate_text[i:i+context_window] for i in range(0, len(candidate_text), context_window)]
    #print(len(candidate_texts))
    if len(candidate_texts) > 0:
        return candidate_texts, "Relevant text found."
    else:
        return candidate_texts, "No relevant text found."

#TODO: migrate to helper function
def search_datasheet(module_name):
    #print(module_name)
    try:
        module_number = re.sub('\D','', module_name)
        #print(module_number)
        candidate_modules = [module_name for module_name in module_names if module_number in module_name] 
        # pick first module for now...
        module_idx = module_names.index(candidate_modules[0])
    except ValueError:
        module_idx = -1
    return f"datasheet{'' if module_idx > -1 else ' not'} found\n""", module_idx
from typing import List, Optional

class ReactAgent:
    observations: List[str] = []
    keywords: List[str] = []
    doc_idx: int = 0
    module_name: str = """"""
    instruction: str = """"""
    examples: str = """"""
    prompt: str = """"""
    answer: str = """"""
    
    def __init__(self,instruction: str, examples: str):
        self.instruction = instruction
        self.examples = examples
        self.reset()
        
    def reset(self):
        self.prompt = self.instruction + self.examples
        self.observations = []
        
        
    @property
    def observation_num(self):
        return len(self.observations) + 1
    
    def append_observation(self,observation):
        observation = f"Observation {self.observation_num}: " + observation
        self.observations.append(observation)
        
    def update_prompt(self, message):
        message = message.replace('\n','')
        #print("Updating prompt with ...")
        print(message)
        self.prompt = self.prompt + '\n' + message + '\n'
    
    def process_response(self, response) -> bool:
        #print(response)
        try:
            self.is_successful = True
            if response.startswith("Action"):
                action = response.split(': ')[-1]
                if action.startswith("Finish"):
                    return True
                elif action.startswith("Search["):
                    self.module_name = action.split('[')[-1][:-1]
                    observation, doc_idx = search_datasheet(self.module_name)
                    self.doc_idx = doc_idx
                elif action.startswith("Annotate"):
                    observation = annotate_question(self.question)
                    self.keywords = observation.split(',')
                elif action.startswith("LookupKeywords"):
                    contexts, observation = lookup_keywords(self.module_name, self.doc_idx, self.keywords)
                    self.contexts = contexts
                elif action.startswith("Reason"):
                    observation = reason_given_context(self.question, self.contexts)
                    self.answer = observation

                self.append_observation(observation)
                self.update_prompt(response)
                self.update_prompt(self.observations[-1])

            elif response.startswith("Thought"):
                self.update_prompt(response)
            
            return False
        
        except Exception as e:
            print(e)
            self.is_successful = False
            return True
        
    def generate_answer(self,question):
        self.question = question
        self.update_prompt(question)
        response = misc_get_completion(self.prompt, stop="\n")
        while not self.process_response(response):
            time.sleep(0.1)
            response = misc_get_completion(self.prompt, stop="\n")
            
        if self.is_successful:
            return self.answer
        else:
            return "I do not know."
    
agent = ReactAgent(instruction=instruction, examples=examples)
agent.generate_answer("What is the recommended range for the value of C1 in the LTC7804 module and why is it chosen in that range?")


