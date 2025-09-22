import logging
from langchain.evaluation.qa import QAEvalChain
from langchain_groq import ChatGroq
from src.config.settings import settings
from src.reranking_genaration import QueryRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("Evaluation")


class Evaluator:
    def __init__(self):
        """Initialize evaluator with LLM (Groq) + QAEvalChain."""
        try:
            self.llm = ChatGroq(
                model_name=settings.groq_model,
                api_key=settings.groq_api_key,
            )
            self.eval_chain = QAEvalChain.from_llm(self.llm)
            self.router = QueryRouter()
            logger.info(f"✅ Evaluation chain initialized with {settings.groq_model}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize evaluator: {e}")
            raise

    def evaluate(self, examples):
        """
        Evaluate RAG system predictions against ground-truth answers.

        Args:
            examples (list[dict]): [{"question": ..., "ground_truth": ...}, ...]

        Returns:
            list[dict]: Evaluation results
        """
        predictions = []

        for ex in examples:
            try:
                prediction = self.router.route(ex["question"])
                predictions.append(
                    {"query": ex["question"], "result": prediction}
                )
                logger.info(f"✅ Prediction generated for: {ex['question']}")
            except Exception as e:
                logger.error(f"❌ Failed to generate prediction for '{ex['question']}': {e}")
                predictions.append({"query": ex["question"], "result": ""})

        formatted_examples = [
            {"query": ex["question"], "answer": ex["ground_truth"]} for ex in examples
        ]

        try:
            graded_outputs = self.eval_chain.evaluate(formatted_examples, predictions)
            return graded_outputs
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return []


if __name__ == "__main__":
    examples = [
        {
            "question": "What is the price and rating of Garlic Oil - Vegetarian Capsule 500 mg?",
            "ground_truth": "Garlic Oil - Vegetarian Capsule 500 mg, from Sri Sri Ayurveda under Beauty & Hygiene → Hair Care, is priced at ₹220 with a customer rating of 4.1.",
        },
        {
            "question": "What are the features of Creme Soft Soap - For Hands & Body by Nivea?",
            "ground_truth": "Nivea Creme Soft Soap, listed under Beauty & Hygiene → Bath & Hand Wash, is a bathing bar enriched with Vitamin F and Almonds. It is priced at ₹162 with a 4.4 rating and is designed to nourish skin while cleansing.",
        },
    ]

    evaluator = Evaluator()
    results = evaluator.evaluate(examples)
    print("\n🔎 Evaluation Results:")
    for r in results:
        print(r)

