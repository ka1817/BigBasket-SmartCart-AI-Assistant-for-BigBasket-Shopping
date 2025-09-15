import logging
from typing import Optional
import warnings
warnings.filterwarnings('ignore')
import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn import metrics
from src.data_ingestion import DataIngestion
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("QueryClassification")

MODEL_DICT = {
    "LogisticRegression": LogisticRegression(class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "NaiveBayes": MultinomialNB()
}

##MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
##mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
##mlflow.set_experiment("Query_Classification_Experiment")

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__)) 
JOBLIB_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(JOBLIB_DIR, exist_ok=True)


class QueryClassifier:
    def __init__(self, test_size: float = 0.3, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None

        self.data_ingestor = DataIngestion()
        self.df: pd.DataFrame = self.data_ingestor.load_data()

        self.df["text"] = (
            self.df["product"].fillna("") + " " +
            self.df["brand"].fillna("") + " " +
            self.df["type"].fillna("") + " " +
            self.df["description"].fillna("")
        )
        self.X = self.df["text"]
        self.y = self.df["category"]

        logger.info(f"QueryClassifier initialized with {self.df.shape[0]} rows.")

    def train(self):
        MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Query_Classification_Experiment")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        logger.info(f"Train/Test split: {X_train.shape[0]} train, {X_test.shape[0]} test")

        for model_name, model in MODEL_DICT.items():
            try:
                logger.info(f"Training model: {model_name}")

                pipeline = Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")),
                    ("clf", model)
                ])

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted")

                logger.info(f"{model_name} - Accuracy: {acc:.4f}, Precision: {precision:.4f}")
                logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

                with mlflow.start_run(run_name=model_name):
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("precision", precision)

                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        name=model_name,  
                        registered_model_name=model_name,
                        input_example=X_test[:2].tolist(),  
                        signature=mlflow.models.infer_signature(X_test, y_test)  
                    )

                joblib_path = os.path.join(JOBLIB_DIR, f"{model_name}_pipeline.joblib")
                joblib.dump(pipeline, joblib_path)
                logger.info(f"{model_name} pipeline saved locally at {joblib_path}")

                logger.info(f"{model_name} model trained, tracked, registered, and saved locally.")
            except Exception as e:
                logger.error(f"Error training model {model_name}: {e}")

    def load_pipeline(self, model_name: str):
        joblib_path = os.path.join(JOBLIB_DIR, f"{model_name}_pipeline.joblib")
        if not os.path.exists(joblib_path):
            raise FileNotFoundError(f"Pipeline file not found: {joblib_path}")

        logger.info(f"Loading pipeline from local file: {joblib_path}")
        self.pipeline = joblib.load(joblib_path)
