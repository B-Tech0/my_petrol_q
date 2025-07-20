from fastapi import FastAPI, Query
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware

import pandas as pd
import os

# Set Groq API Key
os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

# Load data
retail_df = pd.read_csv("PetroData - Retail PetroData.csv")
depot_df = pd.read_csv("PetroData - PetroData Depot.csv")
news_df = pd.read_csv("Petroleum News - Sheet1.csv")

# Create text documents
retail_texts = [
    f"Retail - On {row['Date']} in {row['STATE']} ({row['REGION']}), PMS: ₦{row['PMS']}, AGO: ₦{row['AGO']}, DPK: ₦{row['DPK']}, LPG: ₦{row['LPG']}"
    for _, row in retail_df.iterrows()
]

depot_texts = [
    f"Depot - On {row['Date']} in {row['STATE']} ({row['REGION']}), PMS: ₦{row['PMS']}, AGO: ₦{row['AGO']}, DPK: ₦{row['DPK']}, LPG: ₦{row['LPG']}"
    for _, row in depot_df.iterrows()
]

news_texts = []
for _, row in news_df.iterrows():
    date = row['Date']
    for product in ['PMS', 'AGO', 'DPK', 'LPG']:
        if pd.notna(row[product]):
            news_texts.append(f"News on {product} - {date}: {row[product]}")

# Combine texts and split
all_texts = retail_texts + depot_texts + news_texts
splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.create_documents(all_texts)

# Create vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding)
retriever = db.as_retriever()

# Create QA model
llm = ChatGroq(
    groq_api_key=os.environ["GROQ_API_KEY"],
    model_name="llama3-8b-8192"
)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "Petrol QA is live!"}

@app.post("/ask")
def ask_question(payload: Question):
    answer = qa.run(payload.query)
    return {"question": payload.query, "answer": answer}
