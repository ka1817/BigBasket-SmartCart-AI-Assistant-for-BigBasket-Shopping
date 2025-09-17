import logging
from src.query_classification import QueryClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("RunClassification")

def main():
    try:
        logger.info("Loading existing query classification pipeline...")
        classifier = QueryClassifier()
        classifier.load_pipeline("LogisticRegression")
        logger.info("Query classification pipeline loaded successfully.")

        queries = [
            "chicken curry",
            "Pepsi cold drink"
        ]

        predictions = classifier.pipeline.predict(queries)
        for q, p in zip(queries, predictions):
            print(f"Query: {q} --> Predicted category: {p}")

    except FileNotFoundError as fnf:
        logger.error(f"Model file not found: {fnf}")
    except Exception as e:
        logger.error(f"Failed to load query classification pipeline:{e}")

if __name__ == "__main__":
    main()
