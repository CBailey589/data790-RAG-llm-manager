import os
import shutil
from pathlib import Path
from typing import Callable

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings


class RAGDocumentProcessor:
	'''
	Class for handling raw input documents to prep them for RAG LLM calls. This class:
	- Reads in unprocessed documents using PyPDFLoader
	- Splits the document in accordance with user preferences
	- Generates embeddings for the processed documents
	'''

	def __init__(
		self,
		embedding_model: str
	):
		self.persistent_vectors_dir = "./chromadb"
		self.unprocessed_docs_path = './documents'
		self.processed_docs_path = './processed_documents'
		os.makedirs(self.persistent_vectors_dir, exist_ok=True)
		os.makedirs(self.unprocessed_docs_path, exist_ok=True)
		os.makedirs(self.processed_docs_path, exist_ok=True)

		llm_base_url = os.getenv("LLM_BASE_URL")
		llm_api_key = os.getenv("LLM_API_KEY")
		self.embeddings = OpenAIEmbeddings(
			model=embedding_model,
			base_url=llm_base_url,
			api_key=llm_api_key,
		)

		self.chroma = Chroma(
            persist_directory=self.persistent_vectors_dir,
            embedding_function=self.embeddings,
        )


	def process_new_documents(
		self,
		loader_kwargs: dict = None,
		splitter_chunk_size: int = None,
		splitter_chunk_overlap: int = None,
		splitter_chunk_len_function: Callable = len,
		separators: list = None,
		splitter_kwargs: dict = None
	):
		'''
		Takes unprocessed documents, splits and vectorizes them, and uploads vectors to the local vectorstore.
		'''
		loader_kwargs = loader_kwargs or {}
		splitter_kwargs = splitter_kwargs or {}

		for document_path in Path(self.unprocessed_docs_path).iterdir():
			if not document_path.is_file():
				continue
			if document_path.suffix.lower() != ".pdf":
				print(f"Skipping {document_path}: Can only process .pdf files at this time.")
				continue

			# Load PDF
			print(f"Beginning processing of {document_path}")
			loaded_doc = PyPDFLoader(file_path=str(document_path), **loader_kwargs).load()
			print(f"Loaded {len(loaded_doc)} page(s) from {document_path}")

			# Chunk PDF
			chunk_size = splitter_chunk_size or 1000
			chunk_overlap = splitter_chunk_overlap or 100
			separators = separators or ["\n\n", "\n", ". ", " ", ""]
			doc_splitter = RecursiveCharacterTextSplitter(
				chunk_size=chunk_size,
				chunk_overlap=chunk_overlap,
				length_function=splitter_chunk_len_function,
				separators=separators,
				**splitter_kwargs
			)

			doc_chunks = doc_splitter.split_documents(loaded_doc)
			print(f"{document_path} split into {len(doc_chunks)} chunks")

			# Generate embeddings from document chunks
			self.chroma.add_documents(doc_chunks)

			print("Chroma embedding store updated with new vectors.")

			destination_path = os.path.join(self.processed_docs_path, document_path.name)
			shutil.move(str(document_path), str(destination_path))

			print(f"Finished processing {document_path}")

		print(f"Processing of new documents is complete. Total collection count: {self.chroma._collection.count()} vectors")