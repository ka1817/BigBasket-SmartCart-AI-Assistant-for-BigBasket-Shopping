import logging
import os
import pandas as pd
from src.query_classification import QueryClassifier
from src.config.settings import settings

logger = logging.getLogger("QueryClassificationPipeline")

class QueryClassificationPipeline:
    def run(self, df: pd.DataFrame) -> QueryClassifier:
        logger.info(">>> Running Query Classification Pipeline...")

        classifier = QueryClassifier()

        joblib_path = os.path.join("models", f"{settings.classifier_model}_pipeline.joblib")

        if os.path.exists(joblib_path):
            logger.info(f"Found saved pipeline. Loading {settings.classifier_model}...")
            classifier.load_pipeline(settings.classifier_model)
        else:
            logger.warning("No saved pipeline found. Training a new one...")
            classifier.df = df
            classifier.X = df["product"].fillna("") + " " + df["brand"].fillna("") + " " + df["type"].fillna("") + " " + df["description"].fillna("")
            classifier.y = df["category"]
            classifier.train()
            classifier.load_pipeline(settings.classifier_model)

        return classifier
