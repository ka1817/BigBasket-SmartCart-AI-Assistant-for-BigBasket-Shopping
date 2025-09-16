import logging
import os
import warnings
warnings.filterwarnings('ignore')
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

from src.vectorstore import VectorStoreManager
from src.query_classification import QueryClassifier
from src.config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("QueryRouter")

os.environ["LANGCHAIN_TRACING_V2"] = "true"  
os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


class QueryRouter:
    def __init__(self):
        self.top_k = settings.top_k
        self.confidence_threshold = settings.confidence_threshold

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

        self.llm = ChatGroq(
            model_name=settings.groq_model,
            api_key=settings.groq_api_key
        )
        logger.info(f"✅ LLM loaded: {settings.groq_model}")


        self.prompt = PromptTemplate(
            input_variables=["context", "question", "category"],
            template=settings.rag_prompt)

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

