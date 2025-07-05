

import os
import openai
import time
import numpy as np
import matplotlib.pyplot as plt
import re
import tiktoken

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


    


#Folder Names:
# qas_fname = f"{DATA_DIR}/qas/qas_{dset}.json"
qas_fname = f"{DATA_DIR}/qas/qas_wo_answers_test.json"
qas_dict = read_dict_from_json(qas_fname)
doc_fnames = list(set([DATA_DIR + "/datasheets/" + qa_dict["datasheet"]  for qa_dict in qas_dict.values()])) #parsed/something.txt
context_length = 2 * 4000 # one token is roughly 4 characters (let's be over-conservative)

#Context:
# contexts_fname = f"{DATA_DIR}/qas/contexts_{dset}.json"
contexts_fname = "TextChunks1.json"
contexts_dict = read_dict_from_json(contexts_fname)

#Token Configurations:
MODEL_COST_PER_1K_TOKENS_GPT3P5 = 0.0015
# TOP_K = 1000
# CONTEXT_LENGTH_IN_K_TOKENS = num_tokens * 1e-3
# NUM_QUESTIONS = 1000
# MODEL_COST_PER_1K_TOKENS_GPT3P5 * TOP_K * CONTEXT_LENGTH_IN_K_TOKENS * NUM_QUESTIONS
total_cost_fn = lambda num_tokens: 1e-3 * MODEL_COST_PER_1K_TOKENS_GPT3P5 *  num_tokens 

#Embeddings:
embeddings_fname = f"{DATA_DIR}/qas/embeddings_{dset}.npz" #new trained embeddings, this is not the old!
embeddings_data = np.load(embeddings_fname)
embeddings, texts = embeddings_data["embeddings"], embeddings_data["texts"]

#Vector Database:
if_usingLLMChain = False
if if_usingLLMChain:
    is_generate_vectorDatabase = False
    loaders = [TextLoader(_) for _ in doc_fnames]
    documents = [d for loader in loaders for d in loader.load()]
        # Next we split documents into small chunks. 
        # This is so we can find the most relevant chunks for a query and pass only those into the LLM.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts_chunks = text_splitter.split_documents(documents) 



######## EMBEDDINGS SECTION ###########
is_embeddingFile = Path(embeddings_fname).exists()
print("Embeddings file does" + " exist." if is_embeddingFile else " not exist.")
print("Set `is_generate_embeddingFile` to True if you would like to recompute it.")
is_generate_embeddingFile = False #Set to true to generate Embeddings again

if is_embeddingFile and not is_generate_embeddingFile:
    print("Embedding File exists, loading embeddings from disk")
    # if we have previously created the emebeddings
    # embeddings_data = np.load(embeddings_fname)
    embeddings, texts = embeddings_data["embeddings"], embeddings_data["texts"]

elif is_generate_embeddingFile:
    # let's read all texts and split them into chunks of permissible context length
    texts = []
    for df in doc_fnames:
        text = read_text(df)
        texts += [text[i:i+context_length] for i in range(0, len(text), context_length)]

    embeddings = []
    for text in texts:
        embeddings.append(get_embedding(text))
        # np.savez("embeddingTry.npz", embeddings=embeddings, texts=texts)
    # save
    np.savez(embeddings_fname, embeddings=embeddings, texts=texts)
    # embeddings_data = np.load(embeddings_fname)
embeddings_data = np.load(embeddings_fname) #load npz embedding
######## end of EMBEDDINGS SECTION ###########
# embeddings_text = embeddings_data['texts']
# print(embeddings_text)
# np.savetxt('embeddings_text.csv', embeddings_text, fmt='%s', delimiter=',')
######## VECTORE STORAGE SECTION ###########
if if_usingLLMChain:
    # Create embeddings for each chunk and insert into the Chroma vector database.
    # The persist_directory argument tells ChromaDB where to store the database when it's persisted.
    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = f'{DATA_DIR}/qas/db_{dset}'
    # embedding_function_vectorDB = OpenAIEmbeddings() #TODO: ask yourself, is this necessary? or do we need to load previous embeddings_data["embeddings"] or emeddings from question: get_embeddings(questions)
    embedding_function_vectorDB = embeddings_data
    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


    if Path(persist_directory).exists():
        print("Vector Database exists... loading from disk")
        # Load the Database from disk, and create the chain
        # Be sure to pass the same persist_directory and embedding_function as 
        # you did when you instantiated the database. Initialize the chain we will use for question answering.
        # Now we can load the persisted database from disk, and use it as normal. 
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function_vectorDB)
    else:
        print("Database does not exist. Do you want to recompute it?")
        print("If yes, set `is_generate_vectorDatabase` to True in the next cell.")


    if is_generate_vectorDatabase:
        vectordb = Chroma.from_documents(documents=texts_chunks, embedding=embedding_function_vectorDB, persist_directory=persist_directory)
        vectordb.persist()
    # create the retrieval QA chain with `stuff` mode
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever())

######## end of VECTORE STORAGE SECTION ###########


# prompt = """
# I will give you a piece of text from electrical/electronic module's datashet delimted by ---
# Your task is to give a detailed answer to the question that follows the text based on the provided text.
# If the text does not contain sufficient information to answer, just say I do not know.

# Here is the text:
# ---
# {context}
# ---
# Question: {question}
# Answer:
# """


if True:

    if if_usingLLMChain:
        answers_with_retrieval = {}
        with get_openai_callback() as cb:
            for q_id, qa_dict in (pbar := tqdm(qas_dict.items())):
                pbar.set_description(f"Answering question with call back: {q_id}")
                answer = qa.run(qa_dict["question"])
                answers_with_retrieval[q_id] = answer
            print(cb)
    else: 
        gpt3p5_answers = {}
        max_retries = 1 #was 3
        total_num_tokens = 0
        get_completion_count = 0
        for q_id, qa_dict in (pbar := tqdm(qas_dict.items())):
            
            context = contexts_dict[q_id][0]
            question = qa_dict["question"]
            # for idx, text in enumerate(contexts_dict[q_id]):
            #     if text.rfind(question)> -1:
            #         print(idx)


            prompt = """    
            You are presented with a text excerpt extracted from a datasheet related to electrical or electronic components. The provided text is enclosed within the following delimiters:
            ---
            {context}
            ---

            Your task is to respond to the question using the information from the text. If the text lacks sufficient details for an answer, please indicate "I do not know" and tell why. Approach your response as an engineer addressing a query based on the content of a datasheet. Your answer should be concise and not exceed 2000 characters in length. Additionally, whenever you come across instances of double line breaks ("\n\n") in the text, replace them with a single space in your answer.

            Question: {question}
            Answer:
            """
            if False: 
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                prompt_tokens = encoding.encode(prompt)
                prompt_tokens_length = len(prompt_tokens)
            # print(prompt)
            
            
            context_embeddings = np.array(embeddings_data['embeddings']) 
            
            # for idx, text in enumerate(embeddings_data['texts']):
            #     if text.rfind('C1 value')> -1:
            #         print(idx)
            question_embedding = get_embedding(question) 
            query_embedding = question_embedding
            relatedness_score = context_embeddings.dot(query_embedding)
            top_idxs = np.argsort(-relatedness_score)
            # selected_by_relatedness_context = texts[top_idxs[5]]
            relatedness_score[top_idxs]

            num_retries = 0
            while num_retries < max_retries:
                try:
                    pbar.set_description(f"Answering question {q_id}, Retry: {num_retries}")
                    # for _ in range(len(relatedness_score)):
                    for _ in range(1): #top 11 highest related when range is 10
                        time.sleep(0.2)
                        # gpt3p5_answers[q_id], num_tokens = misc_get_completion(prompt.format(context=get_embedding(context), question=question), is_return_total_tokens=True)
                        gpt3p5_answers[q_id], num_tokens = misc_get_completion(prompt.format(context=embeddings_data['texts'][top_idxs[_]].replace("\n"," ").replace("\n\n"," "), question=question), is_return_total_tokens=True)
                        # gpt3p5_answers[q_id], num_tokens = misc_get_completion(prompt.format(context=OpenAIEmbeddings(), question=question), is_return_total_tokens=True)
                        time.sleep(0.2)
                        total_num_tokens += num_tokens
                        get_completion_count +=1
                        # answer = 'IDK'
                        # print(num_tokens) #accumulate
                        if 'I do not know' not in gpt3p5_answers[q_id]: 
                            print("I know the answer")
                            num_retries = max_retries
                            break
                        else:
                            print("I don not know. Retrieved text does not contain sufficient information.")
                            num_retries +=1
                            continue
                        # print('Answer: ', answer)
                    break
                except Exception as e:
                    num_retries +=1
                    gpt3p5_answers[q_id] = f"I do not know, I have error {e}"
    
        print(f"The total number of token was: {total_num_tokens} \n")
        print(f"Finished answering questions using (USD) ${total_cost_fn(total_num_tokens):.5f} \n")
        print(f"for number of get_completions: {get_completion_count}")
        write_dict_to_json(f"Solving_I_do_not_know_problems.json", gpt3p5_answers)

