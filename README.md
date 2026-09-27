# RAG LLM Manager

## Overview

This project implements a lightweight RAG system for the LLM API calls. The system provides both standard LLM calls and RAG-assisted calls using a local Chroma vector store. It also includes document processing, LLM-based context relevance evaluation, prompt caching, cost tracking, and budget management.

## Architecture

The system is managed through a RAGLLMManger class, which coordinates document processing, LLM calls, RAG retrieval, caching, cost tracking, and budget management. See architecture diagram below.

## Project Structure

src/
- rag_llm_manager.py
- llm_client.py
- rag_llm_client.py
- rag_document_processor.py
- llm_prompt_validator.py
- prompt_cache.py
- cost_tracker.py
- budget_manager.py

documents/             # Unprocessed documents are placed here

processed_documents/   # Dcouments are moved here automatically after processing

chromadb/              # Persistent Chroma vector store

Milestone_1_demo.ipynb # A demonstration of the class and it's capabilities

## Setup

Clone the repository.

Create a Python virtual environment (recommend Python 3.12).

Install the project dependencies using pyproject.toml.

Create a .env file using .env.example and configure the required LLM API settings.

Place source PDF documents in the documents/ directory.

## Usage

Create an instance of the RAGLLMManager class.

Process .PDF documents for use with RAG assisted LLM API calls.

Proceed to make basic and RAG assisted API calls - see Milestone_1_demo.ipynb for examples.

## Features

- PDF document ingestion and chunking
- Embedding generation and Chroma vector storage
- Standard LLM calls
- RAG-assisted LLM calls with LLM-based context relevance evaluation
- Prompt validation
- Response caching
- API usage and cost tracking
- Daily budget management
- Latency and performance tracking

<img width="1725" height="997" alt="image" src="https://github.com/user-attachments/assets/f66f7c58-bde5-440b-8f49-0180c203618f" />
