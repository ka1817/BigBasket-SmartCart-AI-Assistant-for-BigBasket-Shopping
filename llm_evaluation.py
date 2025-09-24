import logging
from pydantic import BaseModel
from langchain_groq import ChatGroq
from src.config.settings import settings
from src.reranking_genaration import QueryRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("Evaluation")


class EvalResult(BaseModel):
    question: str
    prediction: str
    ground_truth: str
    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float
    hallucination_rate: float
    accuracy: float


class Evaluator:
    def __init__(self):
        try:
            self.llm = ChatGroq(
                model_name=settings.groq_model,
                api_key=settings.groq_api_key,
            )
            self.router = QueryRouter()
            logger.info("Custom evaluator ready (LLM-based).")
        except Exception as e:
            logger.error(f"Failed to initialize Evaluator: {e}")
            raise

    def _build_prompt(self, question: str, answer: str, reference: str, retrieved_docs: list[dict]) -> str:
        """Constructs the evaluation prompt for the LLM with 5 metrics."""
        context = "\n\n".join([d.page_content for d in retrieved_docs])
        return f"""
You are an evaluator for a RAG (Retrieval-Augmented Generation) system. 
Given a question, the predicted answer, the ground truth reference, and retrieved documents, 
evaluate the answer along 5 metrics: Recall@K, Precision@K, MRR@K, HallucinationRate, and Accuracy.

Return ONLY a JSON object with numeric scores between 0 and 1. 
Do NOT include any extra text, markdown, or backticks.

Question: {question}

Predicted Answer: {answer}

Ground Truth Reference: {reference}

Retrieved Documents (Top-K): {context}

Evaluation Guidelines (with formulas and examples):

1. Recall@K:
   - Measures the fraction of relevant documents that are retrieved in the top-K results for a query.
   - Formula: Recall@K = (# relevant docs in top-K) / (total relevant docs for that query)
   - Example: A query has 4 relevant documents, top-5 retrieved documents include 2 relevant → Recall@5 = 2/4 = 0.5

2. Precision@K:
   - Measures fraction of retrieved documents that are relevant.
   - Formula: Precision@K = (# relevant docs in top-K) / K
   - Example: Top-5 docs retrieved, 3 are relevant → Precision@5 = 3/5 = 0.6

3. MRR@K (Mean Reciprocal Rank):
   - Measures ranking quality; 1 / rank of first relevant doc.
   - Formula: MRR@K = (1/N) * Σ (1 / rank_i), where rank_i = rank of first relevant doc for query i, N = total queries
   - Example: Query 1 first relevant doc at rank 2, Query 2 at rank 1 → MRR@5 = (1/2 + 1/1)/2 = 0.75

4. HallucinationRate:
   - Fraction of generated statements not supported by reference or retrieved docs.
   - Formula: HallucinationRate = (# unsupported statements) / (total statements in answer)
   - Example: Answer has 10 statements, 2 are unsupported → HallucinationRate = 2/10 = 0.2

5. Accuracy:
   - Measures overlap of predicted answer with ground truth.
   - Formula: 
       - Exact match: Accuracy = 1
       - Partial match: Accuracy = (correct_tokens / total_tokens)
       - Else: Accuracy = 0
   - Example: Reference: "Paris is the capital of France", Predicted: "Paris is capital of France" → 5 tokens correct out of 6 → Accuracy = 5/6 ≈ 0.83

Return JSON exactly in this format:
{{
  "Recall@K": <float>,
  "Precision@K": <float>,
  "MRR@K": <float>,
  "HallucinationRate": <float>,
  "Accuracy": <float>
}}
"""

    def evaluate(self, examples: list[dict]) -> list[EvalResult]:
        """Evaluate dataset examples using custom LLM prompts."""
        results: list[EvalResult] = []

        for ex in examples:
            question = ex["question"]
            ground_truth = ex["ground_truth"]

            try:
                prediction = self.router.route(question)
                if isinstance(prediction, dict):
                    answer = prediction.get("answer", prediction)
                    docs = prediction.get("source_documents", [])
                else:
                    answer = prediction
                    docs = []

                prompt = self._build_prompt(question, answer, ground_truth, docs)

                llm_response = self.llm.invoke(prompt)
                metrics_json = llm_response.content.strip()

                import json
                try:
                    metrics = json.loads(metrics_json)
                except json.JSONDecodeError:
                    logger.warning(f"LLM returned invalid JSON, got: {metrics_json}")
                    continue

                result = EvalResult(
                    question=question,
                    prediction=answer,
                    ground_truth=ground_truth,
                    recall_at_k=metrics.get("Recall@K", 0.0),
                    precision_at_k=metrics.get("Precision@K", 0.0),
                    mrr_at_k=metrics.get("MRR@K", 0.0),
                    hallucination_rate=metrics.get("HallucinationRate", 0.0),
                    accuracy=metrics.get("Accuracy", 0.0),
                )
                results.append(result)
                logger.info(f"Evaluated: {question}")

            except Exception as e:
                logger.error(f"Evaluation failed for '{question}': {e}")

        return results


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

    print("\n🔎 Final Evaluation Metrics:")
    for r in results:
        print(r.model_dump())
