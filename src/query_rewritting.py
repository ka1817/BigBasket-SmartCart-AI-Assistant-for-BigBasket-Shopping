from dotenv import load_dotenv
load_dotenv()

import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def query_rewriting(query: str) -> str:
    llm = ChatGroq(api_key=GROQ_API_KEY, model="gemma2-9b-it")

    prompt = PromptTemplate(
        input_variables=["query"],
        template="""
You are an expert in information retrieval. Your task is to refine the given query to make it **clear, specific, and precise** for searching in a **vector database**.  

### **Instructions:**  
1. **Ensure clarity** – Remove vague or ambiguous terms.  
2. **Be specific** – Include important details for better search results.  
3. **Use a natural, formal structure** suitable for similarity search.  
4. **Do NOT change the original intent of the query.**  

### **Original Query:**  
{query}  

### **Rewritten Query:**  
"""
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query})
    
