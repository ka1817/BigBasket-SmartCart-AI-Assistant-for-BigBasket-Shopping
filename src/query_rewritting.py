import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from src.utils.logger import setup_logger
import sys
from functools import lru_cache


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logger = setup_logger("query_rewriting", "query_rewriting.log")

@lru_cache(maxsize=256)
def query_rewriting(query: str) -> str:
    try:
        logger.info(f"Received query for rewriting: '{query}'")

        llm = ChatGroq(api_key=GROQ_API_KEY, model="gemma2-9b-it")

        prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are an expert in information retrieval. Your task is to refine the given query to make it **clear, specific, and precise** for searching in a **vector database**.

### Instructions:
- Rewrite the query to be natural, formal, and unambiguous.
- Keep the original intent.
- Do NOT return multiple options.
- Do NOT include any explanations.
- Do NOT include headings, bullet points, or formatting.
- Return only the final rewritten query as a single line of text.

Original Query:
{query}

Rewritten Query: 
"""
        )

        chain = prompt | llm | StrOutputParser()
        rewritten_query = chain.invoke({"query": query}).strip()

        logger.info(f"Rewritten query: '{rewritten_query}'")
        return rewritten_query

    except Exception as e:
        logger.error(f"Error during query rewriting: {e}")
        raise

if __name__ == '__main__':
    sample_query = "which is the cheapest hair product with high rating"
    rewritten = query_rewriting(sample_query)
    print(f"Original: {sample_query}\nRewritten: {rewritten}")
