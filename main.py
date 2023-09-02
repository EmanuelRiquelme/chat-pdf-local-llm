from fastapi import FastAPI,Query,BackgroundTasks
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from typing import Annotated
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
import torch
import pickle
#from pathlib import Path

app = FastAPI()
model_id = 'google/flan-t5-large'

qa = None

def initiate_vars():
    global qa 
    try:
        embedding_file =os.path.join(os.path.dirname(__file__),'vectorstore.pkl')
        with open(embedding_file, "rb") as f:
            vectorstore = pickle.load(f)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=100
        )
        local_llm = HuggingFacePipeline(pipeline=pipe)
        prompt_template = """
        You are a sales assistant.
        {context}

        Question: {question}
        Answer here:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm = local_llm,
            memory=memory,
            retriever=vectorstore.as_retriever(),
            combine_docs_chain_kwargs={"prompt": PROMPT},
            )
    except Exception as e:
        print("Error initiating the model:", e)


@app.get("/get_file")
def restful_emb(csv_file: str ,model_name = "BAAI/bge-small-en",background_tasks: BackgroundTasks = None):
    model_kwargs = {'device': 'cuda:0'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    loader = CSVLoader(file_path=csv_file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    documents = text_splitter.split_documents(data)
    vectorstore = FAISS.from_documents(documents, hf)
    with open('vectorstore.pkl', "wb") as f:
        pickle.dump(vectorstore, f)
    background_tasks.add_task(initiate_vars)

@app.get("/query")
async def query(q: str):
    return qa({"question": q})['answer']
