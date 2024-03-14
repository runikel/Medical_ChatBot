from langchain import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import SystemMessagePromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
import os
import json


app = FastAPI()

templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

local_llm = "BioMistral-7B.Q4_K_M.gguf"

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path= local_llm,
    temperature=0.3,
    max_tokens=2048,
    top_p=1,
    n_ctx= 2048
)

print("LLM Initialized....")

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Chat History: {chat_history}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""

embeddings = SentenceTransformerEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")

url = "http://localhost:6333"

client = QdrantClient(
    url=url, prefer_grpc=False
)

db = Qdrant(client=client, embeddings=embeddings, collection_name="vector_db")


retriever = db.as_retriever(search_kwargs={"k":1})

chat_history = []


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/get_response")
# async def get_response(query: str = Form(...)):
#     chain_type_kwargs = {"prompt": prompt}
#     qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
#     response = qa(query)
#     print(response)
#     answer = response['result']
#     source_document = response['source_documents'][0].page_content
#     doc = response['source_documents'][0].metadata['source']
#     response_data = jsonable_encoder(json.dumps({"answer": answer, "source_document": source_document, "doc": doc}))
    
#     res = Response(response_data)
#     return res


@app.post("/get_response")
async def get_response(query: str = Form(...)):
    # Create the custom chain
    if llm is not None and db is not None:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=retriever, 
            # memory=memory,
            # get_chat_history=chat_history, 
            # return_source_documents=True,
            # combine_docs_chain_kwargs={'prompt': prompt}
        )
    else:
        print("LLM or Vector Database not initialized")

    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", 'question'])


    # chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(prompt)
    response = chain({"question": query, "chat_history": chat_history})
    print(response)
    answer = response['answer']
    chat_history.append((query, answer))
    # source_document = response['source_documents'][0].page_content
    # doc = response['source_documents'][0].metadata['source']
    response_data = jsonable_encoder(json.dumps({"answer": answer}))
    
    res = Response(response_data)
    return res