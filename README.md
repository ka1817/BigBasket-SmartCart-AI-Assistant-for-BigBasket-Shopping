## 🛒 BigBasket SmartCart – AI Assistant for BigBasket Shopping
---
##  Introduction

The rapid evolution of AI technologies has created new opportunities for enhancing user experience in digital commerce. Leveraging state-of-the-art language models and retrieval systems, intelligent assistants can now understand complex queries, process vast amounts of product data, and deliver precise, context-aware responses. This project presents a scalable and robust AI-powered shopping assistant tailored for BigBasket's product ecosystem. Built using Retrieval-Augmented Generation (RAG), vector embeddings, and large language models (LLMs), the system enables efficient and intelligent product discovery through natural language interaction.

---

## ❗ Problem Statement

Online shoppers frequently seek personalized and context-specific product recommendations, such as identifying the best-rated skincare item at the lowest price. However, conventional search systems often fall short in understanding such nuanced queries, lacking the ability to interpret intent, compare attributes across products, and deliver concise, relevant results. This creates friction in the user journey, leading to suboptimal shopping experiences. There is a clear need for an intelligent assistant that can process natural language queries, reason over structured product data, and deliver accurate, insightful responses to aid decision-making.

---

## Business Goal :

To enhance the shopping experience, boost conversion rates, and optimize search efficiency by enabling natural language-based product search that understands user intent and delivers context-aware, personalized recommendations.



## 💰 Business Impact (Revenue + Cost)

💸 1. Increased Conversion Rates (↑ Revenue)

    • Users find relevant products faster, leading to more product views, cart adds, and purchases

    • Personalized recommendations match buyer intent better than traditional search

    • Better UX = lower drop-off rates

📈 Even a 1–2% uplift in conversions from improved product search can lead to significant revenue gains for a large marketplace like BigBasket.

📉 2. Reduced Customer Support Queries (↓ Cost)

    • AI assistant can handle informational and product-related queries

    • Reduces manual intervention, live chat support, and email volume

    • More self-service = less operational overhead

⏱️ 3. Reduced Time-to-Purchase (↑ Efficiency)

    • Customers make faster decisions because the assistant summarizes comparisons (e.g., price vs. rating trade-offs)

    • This shortens the purchase journey and increases user satisfaction

🧪 4. Rapid Experimentation & Deployment (↓ Dev Costs)

    • The project is modular, Dockerized, and CI/CD enabled → easier to iterate and deploy

    • Can be extended to other verticals (electronics, fashion) or other marketplaces with minimal changes

---

## 🚀 Features

Here’s the refined **architecture diagram** for your BigBasket RAG pipeline and a detailed breakdown of its **features**:

---

## 🏗️ Correct Architecture Diagram

```
                   ┌──────────────┐
                   │   User Query │
                   └──────┬───────┘
                          │
                          ▼
                  ┌───────────────────┐
                  │ Query Classifier  │
                  │ (LogisticRegression) │
                  └─────────┬─────────┘
                            │ Predicted Category + Confidence
                            ▼
          ┌─────────────────┴──────────────────┐
          │                                    │
Confidence ≥ Threshold                 Confidence < Threshold
          │                                    │
          ▼                                    ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│ Category-Specific Retriever│      │ Global Retriever          │
│ (Pinecone VectorStore)     │      │ (Pinecone VectorStore)    │
│ + ContextualCompression    │      │ + ContextualCompression   │
│ + Cross-Encoder Reranker   │      │ + Cross-Encoder Reranker  │
└───────────────┬───────────┘      └───────────────┬───────────┘
                │                                  │
                ▼                                  ▼
       Retrieved & Reranked Docs          Retrieved & Reranked Docs
                │                                  │
                └──────────────┬───────────────────┘
                               ▼
                        ┌───────────────┐
                        │   RAG Prompt  │
                        │ (context,     │
                        │ question, cat)│
                        └──────┬────────┘
                               ▼
                        ┌───────────────┐
                        │     LLM       │
                        │   ChatGroq    │
                        └──────┬────────┘
                               ▼
                       ┌────────────────┐
                       │  Final Answer  │
                       └────────────────┘
```
---

## ✨ Features of This Project

### 🔹 Data Layer

* **Data Ingestion**: Loads BigBasket product data (CSV) with logging & error handling.
* **Preprocessing**: Cleans missing values, enforces numeric conversions, and enriches data with `category_tag`.
* **Document Creation**: Converts product data into `LangChain Document` objects for RAG.

### 🔹 Query Understanding

* **Query Classification**:

  * Uses **Logistic Regression**/**Random Forest** with TF-IDF features.
  * Trained with category labels to classify user queries.
  * MLflow integration for experiment tracking & model registry.
  * Models persisted locally with Joblib.

### 🔹 Vector Database & Retrieval

* **Pinecone VectorDatabase**:

  * Stores product embeddings using HuggingFace embeddings.
  * Supports both **global retrieval** (across all categories) and **category-specific retrieval** (filtered by classifier).
* **Retriever Types**:

  * **Category Retriever** → Activated if classifier confidence ≥ threshold.
  * **Global Retriever** → Activated if classifier confidence < threshold.

### 🔹 Reranking

* **ContextualCompressionRetriever**:

  * Uses a **cross-encoder (MiniLM)** for reranking retrieved results.
  * Ensures most relevant documents are prioritized before passing to the LLM.

### 🔹 Generation

* **RAG Prompt**:

  * Dynamically structured with context, question, and predicted category.
* **ChatGroq LLM**:

  * Provides natural language answers.
  * Powered by Groq API for efficient text generation.

### 🔹 Logging & Monitoring

* Structured logging at every stage (data loading, preprocessing, training, retrieval, reranking, generation).
* Error handling ensures resilience in pipeline execution.

### 🔹 Scalability & Modularity

* Separate modules for:

  * `data_ingestion.py`
  * `data_preprocessing.py`
  * `query_classification.py`
  * `vectorstore.py`
  * `retrieval_generation.py`
  * `reranking_generation.py`
* Easy to extend with new retrievers, embeddings, or classifiers.


---

## 🗂️ Folder Structure

Here’s the **project structure** based on your screenshots:

```
BIGBASKET-RAG/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD pipeline config
│
├── .pytest_cache/                 # Pytest cache files
├── data/                          # Dataset directory (e.g., BigBasket Products.csv)
├── mlartifacts/                   # MLflow artifacts
├── mlruns/                        # MLflow experiment runs
├── models/                        # Trained model pipelines (Joblib)
├── notebook/                      # Jupyter notebooks for experimentation
├── scripts/                       # Utility or automation scripts
│
├── src/                           # Main source code
│   ├── __pycache__/               # Python cache files
│   ├── config/                    # Configuration settings
│   │   └── settings.py
│   ├── __init__.py
│   ├── data_ingestion.py          # Loads product data from CSV
│   ├── data_preprocessing.py      # Cleans data and prepares LangChain Documents
│   ├── query_classification.py    # ML models for query classification
│   ├── reranking_generation.py    # QueryRouter with reranking using cross-encoder
│   ├── retrival_generation.py     # QueryRouter without reranking
│   └── vectorstore.py             # Pinecone VectorStore manager
│
├── static/                        # Static assets (CSS)
│   └── css/
│
├── templates/                     # Frontend templates (HTML, Jinja2)
│   └── index.html
│
├── tests/                         # Unit and integration tests
├── ui/                            # UI image
├── vectorstore/                   # Additional vectorstore-related scripts/configs
├── venv/                          # Python virtual environment (ignored in Git)
│
├── .dockerignore                  # Ignore rules for Docker
├── .env                           # Environment variables (API keys, configs)
├── .gitignore                     # Ignore rules for Git
├── Dockerfile                     # Docker image setup
├── main.py                        # Application entry point
├── pytest.ini                     # Pytest configuration
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

---

## 🧪 Local Development Setup

```bash
# Clone the repository
git clone https://github.com/ka1817/BigBasket-SmartCart-AI-Assistant-for-BigBasket-Shopping
cd BigBasket

# Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Before Running the app set .env(environment variable GROQ_API_KEY)
uvicorn main:app --reload --port 8000
```

## 🐳 Docker Instructions

🔧 1. Pull Image

```bash
docker pull pranavreddy123/bigbasket-assistant:latest
```

🚀 2. Run the App (Detached Mode)

```bash
docker run -d -p 8000:8000 \
-e GROQ_API_KEY=create groq api from groq cloud \
pranavreddy123/bigbasket-assistant:latest
```

🌐 3. Access the App

```bash
http://localhost:8000
```

---

## 🛠️ GitHub Actions (CI/CD)

File: .github/workflows/ci-cd.yml

✅ CI-Test: Runs unit tests using pytest.

🐳 CD-Docker: Builds Docker image and pushes to DockerHub.

Triggered on push to main or pull request.

---

## ☁️ Deployment on Amazon EC2

### 1. Launch EC2 Instance (Ubuntu 20.04)

### 2. SSH into your instance

```bash
ssh -i "your-key.pem" ubuntu@your-ec2-ip
```

### 3. Install Docker

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

### 4. Pull and Run Docker Image

```bash
docker pull pranavreddy123/bigbasket-assistant:latest
# Ensure your .env file is in the same directory, or create an API key using Groq Cloud and add it to the .env file
docker run -d --env-file .env -p 8000:8000 pranavreddy123/bigbasket-assistant:latest
```

## Access your app via `http://<your-ec2-public-ip>`

## 🧠 Tech Stack

✅ LLMs: Groq (llama3-70b-8192)

✅ LangChain,Pinecone ,FAISS(experimentation), HuggingFace, CrossEncoder

✅ FastAPI

✅ Docker

✅ GitHub Actions

✅ AWS EC2

✅ HTML/CSS

---

## 🔗 Links

🔍 GitHub Repo: BigBasket-SmartCart-AI-Assistant-for-BigBasket-Shopping

🐳 DockerHub: pranavreddy123/bigbasket-assistant

---

## 🧑‍💻 Developed By

Pranav Reddy
