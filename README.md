# RAG LLM Manager

## Overview

This project provides a lightweight RAG system that can be used to search against the full 285 issue run of Nintendo Power magazine.

The Milestone 2 implementation uses a 285 issue Nintendo Power magazine corpus to answer questions about historical Nintendo Power content. It adds security checks for user prompts and retrieved documents, attempts to identify and redact personally identifiable information (PII), and evaluates generated answers for relevance and adequacy.

## Architecture

The system is managed through a RAGLLMManger class, which coordinates document processing, LLM calls, RAG retrieval, caching, cost tracking, and budget management.

The project includes two RAG approaches for comparison:
- Traditional RAG: Retrieves a fixed number of documents and passes them to the response-generation model.

- Self-RAG-inspired approach: Uses an additional LLM-based evaluation step to assess prompts, retrieved documents, and generated answers. It can filter unsafe or irrelevant context and retry retrieval with additional documents when an answer is insufficient.

Security checks are applied to user prompts and retrieved documents before response generation. The system also attempts to identify and redact PII in supporting context and generated responses.

## Project Structure

src/
- rag_llm_manager.py
- rag_llm_client.py
- rag_document_processor.py
- llm_prompt_validator.py
- prompt_cache.py
- cost_tracker.py
- budget_manager.py

chromadb_db_nintendo_power/              # Persistent Chroma vector store containing embeddings of Nintendo Power 

Milestone_2_retry.ipynb # A demonstration of the security features of the system

## Corpus

The Milestone 2 corpus consists of Nintendo Power magazine content processed into two-page spreads. Each spread is stored as a separate document for retrieval.

The vector store contains 17,309 documents:

17,303 Nintendo Power spreads

6 synthetic security-test documents used to evaluate handling of malicious retrieved content

The project uses the text-embedding-3-large embedding model to generate document embeddings for ChromaDB.

## Setup

Clone the repository.

Create a Python virtual environment (recommend Python 3.12).

Install the project dependencies using pyproject.toml.

Create a .env file using .env.example and configure the required LLM API settings. Utilize text-embedding-3-large for text embedding.

Configure access to the LLM and embedding models used by the project.

Ensure the corpus and persistent ChromaDB data are available before running retrieval experiments.

To build the corpus, download the HOCR .zip file from: https://archive.org/details/nintendo-power-magazines/Nintendo%20Power%20Issue%20001/page/61/mode/2up
and place it in your cloned directory. Unzip the files into a directory named documents/

Follow the np_corpus_prep notebook to locally build your required corpus.

## Usage

Create an instance of the RAG client:
```
import os
from dotenv import load_dotenv
from src.rag_llm_client import RAGLLMClient

load_dotenv()

client = RAGLLMClient(
    embedding_model=os.getenv("DEFAULT_EMBEDDINGS_MODEL"),
    model=os.getenv("DEFAULT_MODEL")
)
```

Ensure the chromadb has all intended documents:
```
# expecting 17,309: 17,303 legitimate + 6 security test document:
print("Stored embeddings:", client.collection.count())
print("Collection:", client.collection.name)
```

Make a call against the Nintendo Power corpus:
```
client.call(
    query='What are some techniques that can help me in Terranigma?',
    k=3,
    self_rag=True,
    security_enabled=True,
    verbose=True
)
```

## Features

- RAG-assisted LLM calls with LLM-based context relevance evaluation
- Basic and Self-RAG approaches
- Prompt validation and LLM based security screening
- Screening of supporting documents for malicious instructions
- Answer relevance evaluation with limited retry 
- PII identification and redacting
- Security test suite

## Dataset Attribution

This project uses OCR-derived text from digitized issues of *Nintendo Power*, originally published by Nintendo of America.

**Digitized source:**
- Internet Archive: https://archive.org/details/nintendo-power-magazines/Nintendo%20Power%20Issue%20001/page/61/mode/2up
- Accessed: October 2026

**Processing and use:**

The original digitized magazine pages were processed into OCR-derived text, converted into structured documents, and divided into page-based chunks for retrieval and evaluation in the RAG system.

Each processed document retains source metadata, including issue identifiers and page references, to support source attribution and traceability.

This project is an academic demonstration of retrieval-augmented generation.
