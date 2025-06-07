# src/retrival_generation.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.data_preprocessing import preprocess_bigbasket_docs
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import warnings
warnings.filterwarnings('ignore')

# Load GROQ API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def ingest_bigbasket_data():
    """Preprocess and embed product docs, store in FAISS vectorstore."""
    docs = preprocess_bigbasket_docs()
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
    vectorstore = FAISS.from_documents(docs, embedding_model)
    
    return vectorstore


def generate_bigbasket_chain(vectorstore):
    """Create RAG chain for answering product queries using filtered retriever."""
    
    # 🔍 FILTERED RETRIEVAL
    retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5  
    }
)


    PRODUCT_BOT_TEMPLATE = """
    You are a helpful assistant that provides answers to user queries based on BigBasket product information.

    Instructions:
    - Base your answer only on the provided context.
    - Include details like price, rating, and brand.
    - Summarize multiple products if they match.
    - Say "Sorry, no matching product information was found in the dataset." if nothing matches.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    """

    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)
    llm = ChatGroq(api_key=GROQ_API_KEY, model="llama3-70b-8192")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
if __name__ == '__main__':
    vstore = ingest_bigbasket_data()

    chain =generate_bigbasket_chain(vstore)

    question = "which product has the highest rating"
    response = chain.invoke(question)

    print(f"Response: {response}")