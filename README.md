## 🛒 BigBasket SmartCart – AI Assistant for BigBasket Shopping

🧾 Introduction

The rapid evolution of AI technologies has created new opportunities for enhancing user experience in digital commerce. Leveraging state-of-the-art language models and retrieval systems, intelligent assistants can now understand complex queries, process vast amounts of product data, and deliver precise, context-aware responses. This project presents a scalable and robust AI-powered shopping assistant tailored for BigBasket's product ecosystem. Built using Retrieval-Augmented Generation (RAG), vector embeddings, and large language models (LLMs), the system enables efficient and intelligent product discovery through natural language interaction.

---

❗ Problem Statement
Online shoppers frequently seek personalized and context-specific product recommendations, such as identifying the best-rated skincare item at the lowest price. However, conventional search systems often fall short in understanding such nuanced queries, lacking the ability to interpret intent, compare attributes across products, and deliver concise, relevant results. This creates friction in the user journey, leading to suboptimal shopping experiences. There is a clear need for an intelligent assistant that can process natural language queries, reason over structured product data, and deliver accurate, insightful responses to aid decision-making.

---

🚀 Features

🔍 Natural Language Product Search
Users can ask queries like "cheapest skin care with highest rating" or "best perfume under ₹500".

🧠 Query Rewriting with LLM
Uses Groq LLMs (gemma2-9b-it) to refine user queries for more precise retrieval.

📄 Document Embedding & Vector Search
Preprocessed BigBasket product data embedded with thenlper/gte-small and indexed using FAISS.

🤖 RAG Pipeline
Uses llama3-70b-8192 model for final answer generation based on retrieved and reranked results.

🔁 Reranking with CrossEncoder
Improves accuracy using cross-encoder/ms-marco-MiniLM-L-6-v2.

🌐 FastAPI Backend
Easily accessible via localhost:8000 or deployed server.

🐳 Dockerized
Build once, run anywhere. Fully containerized using Docker.

🚰 CI/CD with GitHub Actions
Automated testing, image build, and push to DockerHub.

📜 Logging
Logging implemented for each step in the pipeline for transparency and debugging.

---

## 🗂️ Folder Structure

```bash
BIGBASKET/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── data/
│   └── BigBasket Products.csv
├── logs/
│   ├── data_ingestion.log
│   ├── data_preprocessing.log
│   ├── query_rewriting.log
│   └── retrieval_generation.log
├── src/
│   ├── utils/
│   │   └── logger.py
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── query_rewritting.py
│   └── retrival_genaration.py
├── static/
│   └── css/
│       └── style.css
├── templates/
│   └── index.html
├── tests/
├── ui/
├── main.py
├── Dockerfile
├── requirements.txt
├── .env
├── .dockerignore
├── .gitignore
└── README.md
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

🤖 Example Usage

Query: "Which is the cheapest hair product with high rating?"
Rewritten: "Find the most affordable hair care product with a high customer rating."
Response: "Garlic Oil - Vegetarian Capsule 500 mg by Sri Sri Ayurveda is available at ₹220 with a 4.1 rating."

---

## 🛠️ GitHub Actions (CI/CD)

File: .github/workflows/ci-cd.yml

✅ ci-test: Runs unit tests using pytest.

🐳 cd-docker: Builds Docker image and pushes to DockerHub.

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

✅ LLMs: Groq (gemma2-9b-it, llama3-70b-8192)

✅ LangChain, FAISS, HuggingFace, CrossEncoder

✅ FastAPI

✅ Docker

✅ GitHub Actions

✅ AWS EC2

✅ HTML/CSS

---

🔗 Links

🔍 GitHub Repo: BigBasket-SmartCart-AI-Assistant

🐳 DockerHub: pranavreddy123/bigbasket-assistant

---

🧑‍💻 Author
Pranav Reddy
