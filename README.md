# RAG LLM Manager

## Overview

This project implements a lightweight RAG system for the LLM API calls. The system provides both standard LLM calls and RAG-assisted calls using a local Chroma vector store. It also includes document processing, LLM-based context relevance evaluation, prompt caching, cost tracking, and budget management.

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
- llm_client.py
- rag_llm_client.py
- self_rag_llm_client.py
- rag_document_processor.py
- llm_prompt_validator.py
- prompt_cache.py
- cost_tracker.py
- budget_manager.py

documents/             # Unprocessed documents are placed here

processed_documents/   # Dcouments are moved here automatically after processing

chromadb/              # Persistent Chroma vector store

Milestone_1_demo.ipynb # A demonstration of the class and it's capabilities
Milestone_2_evaluation.ipynb # A demonstration of the improvements made for Milestone 2
np_corpus_prep.ipynb # A walkthrough of Nintendo Power corpus generation

## Corpus

The Milestone 2 corpus consists of Nintendo Power magazine content processed into two-page spreads. Each spread is stored as a separate document for retrieval.

The vector store contains 17,304 documents:

17,303 Nintendo Power spreads

1 synthetic security-test document used to evaluate handling of malicious retrieved content

The project uses the text-embedding-3-large embedding model to generate document embeddings for ChromaDB.

## Setup

Clone the repository.

Create a Python virtual environment (recommend Python 3.12).

Install the project dependencies using pyproject.toml.

Create a .env file using .env.example and configure the required LLM API settings.

Configure access to the LLM and embedding models used by the project.

Ensure the corpus and persistent ChromaDB data are available before running retrieval experiments.

## Usage

Create an instance of the RAGLLMManager class.

See the included notebooks and source code for examples of initialization, document processing, querying, and evaluation.
- Specifically Milestone_2_Evaluation.py, specifically the 2 final cells of the ```Define evaluation framework``` section.

#### Recommended test prompts:
- In the original Pikmin how much time do you have to escape the planet? And what is the difference between the Red, Yellow, and Blue Pikmin?

- What tips does Nintendo Power provide for playing Super Bowser: Bowser's Conquest on Game Boy, including the encounter with Mario?

## Features

- Document ingestion and chunking
- Embedding generation and Chroma vector storage
- Standard LLM calls
- RAG-assisted LLM calls with LLM-based context relevance evaluation
- Basic and Self-RAG approaches
- Prompt validation and LLM based security screening
- Screening of supporting documents for malicious instructions
- Answer relevance evaluation with limited retry 
- PII identification and redacting
- Security test suite

<img width="1725" height="997" alt="image" src="https://github.com/user-attachments/assets/f66f7c58-bde5-440b-8f49-0180c203618f" />
