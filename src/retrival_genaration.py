import logging
import os
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

from src.vectorstore import VectorStoreManager
from src.query_classification import QueryClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("QueryRouter")

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


class QueryRouter:
    def __init__(self, top_k: int = 5, confidence_threshold: float = 0.5):
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        self.classifier = QueryClassifier()
        try:
            self.classifier.load_pipeline("LogisticRegression")
            logger.info("✅ Classifier pipeline loaded successfully.")
        except FileNotFoundError:
            logger.warning("No classifier pipeline found. Training a new one...")
            self.classifier.train()
            self.classifier.load_pipeline("LogisticRegression")
            logger.info("✅ Classifier trained and loaded successfully.")

        self.vs_manager = VectorStoreManager()

        try:
            vs = self.vs_manager.load_vectorstore()
            self.global_retriever = vs.as_retriever(search_kwargs={"k": self.top_k})
            logger.info("✅ Global retriever initialized.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize global retriever: {e}")
            self.global_retriever = None

        self.retrievers = {}
        all_categories = self.classifier.y.unique()
        for category in all_categories:
            try:
                vs = self.vs_manager.load_vectorstore()
                self.retrievers[category] = vs.as_retriever(search_kwargs={
                    "k": self.top_k,
                    "filter": {"category": category}
                })
                logger.info(f"✅ Retriever cached for category: {category}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize retriever for '{category}': {e}")

        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        logger.info("✅ LLM loaded successfully.")

        self.prompt = PromptTemplate(
            input_variables=["context", "question", "category"],
            template="""
You are a helpful assistant.

Category: {category}

Use the context below to answer the question. Only use the information from the context.
If the context does not provide enough info, say "I don't know."

Context:
{context}

Question:
{question}

Answer clearly and concisely:
            """,
        )

    def route(self, query: str) -> dict:
        probs = self.classifier.pipeline.predict_proba([query])[0]
        predicted_idx = probs.argmax()
        predicted_category = self.classifier.pipeline.classes_[predicted_idx]
        confidence = probs[predicted_idx]

        logger.info(f"Predicted category: {predicted_category} (confidence={confidence:.2f})")

        if confidence < self.confidence_threshold:
            logger.warning("⚠️ Low confidence. Using global retriever.")
            retriever = self.global_retriever
            effective_category = "All Categories"
        else:
            retriever = self.retrievers.get(predicted_category)
            effective_category = predicted_category

        if retriever is None:
            logger.error(f"No retriever available for category '{effective_category}'")
            return {
                "category": effective_category,
                "confidence": float(confidence),
                "answer": "No vector store available for this category.",
                "source_documents": []
            }

        rag_chain = (
            RunnableParallel({
                "context": RunnableLambda(lambda inputs: retriever.invoke(inputs["question"]))
                          | RunnableLambda(lambda docs: "\n\n".join([d.page_content for d in docs])),
                "question": RunnablePassthrough(),
                "category": RunnablePassthrough()
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke({
            "question": query,
            "category": effective_category
        })
        return answer

