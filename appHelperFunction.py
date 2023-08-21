
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



def get_completion(prompt: str, is_chat_model=True) -> str:
    
    if is_chat_model:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= messages,
            temperature=0
        )
        return response.choices[0].message["content"]
    else:
        response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        #stop="\n"
        )
        return response["choices"][0]["text"]
    

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return np.array(openai.Embedding.create(input = [text], model=model)['data'][0]['embedding'])




def annotate_question(question):
    prompt = """
    I will give you a question about an electrical/electronic module's datashet delimted by ---
    Your task is to generate a list of keywords that best annotates the question
    
    Here is an example:
    Question: What is the purpose of the Conditional Search ROM command in the DS28E04-100 module
    Keywords: Conditional Search, ROM, command

    Here is the question:
    ---
    {question}
    ---
    Keywords:
    """
    response = get_completion(prompt.format(question=question), stop="\n")  
    return response


def translate_response(response):
    prompt = """
    I will give you a piece of text that represents an answer to a question.
    I want you to reply with one letter either Y or N.
    Y - if the text indicates that the question cant be answered.
    N - if the text represents a potential answer to the question.
    
    Here is the text
    ```
    {text}
    ```
    output Y or N as outlined earlier:
    """
    is_idk = get_completion(prompt.format(text=response), stop="\n")
    #print(response)
    #print(f"translating response: is_idk={is_idk}")
    if is_idk == 'Y' or 'IDK' in response:
        return 'I do not know.'
    else:
        return response

def reason_given_context(question, contexts):
    prompt = """
    I will give you a peice of text from electrical/electronic module's datashet delimted by ---
    Your task is to either to give a detailed answer to the question that follows the text 
    based on the provided text or say `IDK` if the text does not contain sufficient information.
    Here is the text:
    ---
    {text}
    ---
    Question: {question}
    Answer/Output `IDK` if you cant answer:
    """
    for idx, context in enumerate(contexts):
        print(f"reasoning attempt {idx}")
        response = get_completion(prompt.format(text=context.replace("\n"," "), question=question), stop="\n")
        response = translate_response(response)
        #print(response)
        if 'I do not know.' not in response: break
        if idx > 10: break
    return response
    


