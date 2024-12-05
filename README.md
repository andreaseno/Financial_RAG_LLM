# Financial Chatbot with RAG and Ollama

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Docker Setup](#docker-setup)
7. [Database Setup](#database-setup)
8. [Running the Chatbot](#running-the-chatbot)
9. [Contributing](#contributing)
10. [License](#license)

## Project Overview
This project is a financial chatbot leveraging Retrieval-Augmented Generation (RAG) with Ollama as the large language model (LLM). It retrieves financial data from a database of SEC 10-Q filings and uses hybrid search (vector similarity + keyword search) to provide contextually relevant answers.

## Features
- **Natural Language Financial Queries:** Users can ask questions about company filings, and the chatbot responds with relevant data from 10-Q forms.
- **Hybrid Retrieval System:** Combines keyword and vector similarity search using pgvector.
- **RAG Framework:** Augments the LLM's capabilities with database-backed factual information for accurate responses.
- **Dockerized Setup:** Ensures easy deployment and database persistence using Docker.

## Tech Stack
- **Backend:** Python
- **Frontend:** CLI (for right now)
- **LLM:** Ollama
- **Database:** PostgreSQL with `pgvector` extension
- **Containerization:** Docker

## Installation
### Prerequisites
- Docker installed
- Homebrew installed
- Ollama setup on macOS
- Python 3.8+ installed

### Clone the Repository
```bash
git clone https://github.com/yourusername/financial-chatbot
cd financial-chatbot
```

### Setup
docker pull pgvector/pgvector:pg16

docker run --name pgvector-container \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=adminpass \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  -v pgvector-data:/var/lib/postgresql/data \
  -d pgvector/pgvector:pg16


### Help
To access DB from CLI, run:

docker exec -it pgvector-container psql -U admin -d vectordb




