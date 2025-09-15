from src.query_classification import QueryClassifier
from src.query_classification import QueryClassifier

if __name__ == "__main__":
    classifier = QueryClassifier()

    classifier.train()

    classifier.load_pipeline("LogisticRegression")

    queries = [
        "chicken curry",    
        "Pepsi cold drink" 
    ]

    predictions = classifier.pipeline.predict(queries)
    for q, p in zip(queries, predictions):
        print(f"Query: {q} --> Predicted category: {p}")
