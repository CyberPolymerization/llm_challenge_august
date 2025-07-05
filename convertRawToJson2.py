
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
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.llms import HuggingFaceHub

set_openai_api_key()
DATA_DIR = 'data'
dset = 'tutorial'


qas_fname = f"{DATA_DIR}/qas/qas_wo_answers_test.json"
qas_fname_parsed = f"{DATA_DIR}/datasheets/raw/"
qas_dict = read_dict_from_json(qas_fname)
doc_fnames = list(set([DATA_DIR + "/datasheets/" + qa_dict["datasheet"]  for qa_dict in qas_dict.values()])) #parsed/something.txt

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    # text_splitter = CharacterTextSplitter(
    #     separator="\n",
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     length_function=len
    # )
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20000, chunk_overlap=0
    )
    chunks = text_splitter.split_text(text)
    return chunks

result_dict = {}
for q_id, qa_dict in (pbar := tqdm(qas_dict.items())):
    # print(qa_dict["question"])
    # print(qa_dict["answer"])
    readyToStrip = qas_fname_parsed+qa_dict["datasheet"]
    pdf_doc = readyToStrip.replace("parsed/", "/").replace(".txt", ".pdf")
    texts = get_pdf_text(pdf_doc)
    text_chunks = get_text_chunks(texts)
    result_dict[q_id] = text_chunks
write_dict_to_json(f"TextChunks1.json", result_dict)

