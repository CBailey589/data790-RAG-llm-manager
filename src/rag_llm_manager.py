import os
from typing import Callable

from dotenv import load_dotenv

from .llm_client import LLMClient
from .rag_document_processor import RAGDocumentProcessor
from .rag_llm_client import RAGLLMClient


class RagLlmManager:
	'''
	Main Class for managing RAG-based LLM interaction. This class provides:
	-
	-
	-
	'''

	def __init__(
		self,
		model: str = None,
		embeddings_model: str = None
	):
		'''
		Initializes RAG LLM Manager and required classes.
		'''
		load_dotenv()

		llm_base_url = os.getenv("LLM_BASE_URL")
		assert llm_base_url, "Missing LLM_BASE_URL — please ensure .env file has the LLM_BASE_URL field populated"
		llm_api_key = os.getenv("LLM_API_KEY")
		assert llm_api_key, "Missing LLM_API_KEY — please ensure .env file has the LLM_API_KEY field populated"
		default_model = os.getenv("DEFAULT_MODEL")
		assert default_model, "Missing DEFAULT_MODEL — please ensure .env file has the DEFAULT_MODEL field populated"
		default_embeddings_model = os.getenv("DEFAULT_EMBEDDINGS_MODEL")
		assert default_embeddings_model, "Missing DEFAULT_EMBEDDINGS_MODEL — please ensure .env file has the DEFAULT_EMBEDDINGS_MODEL field populated"

		self.llm_base_url = llm_base_url
		self.llm_api_key = llm_api_key
		self.model = model or default_model
		self.default_embeddings_model = default_embeddings_model

		self.llm_client = LLMClient()

		embeddings_model = embeddings_model or self.default_embeddings_model
		self.rag_document_processor = RAGDocumentProcessor(embeddings_model)
		self.rag_llm_client = RAGLLMClient(embedding_model=embeddings_model, model=self.model)

	def set_llm_model(self, model: str, **kwargs):
		'''
		Allows the user to change the LLM model (ex: 'gpt-4.1-mini', 'gpt-5.6-sol')
		'''
		self.model = model
		self.rag_llm_client.update_llm_model(model=self.model, **kwargs)
		print(f"UPDATED LLM MODEL TO: {model}")

	def basic_llm_call(self, messages: list, **kwargs):
		'''
		Allows the user to conduct a basic LLM call.
		Returns a BasicLLMResponse object consisting of:
		- success: boolean
		- response: response object from the API
		'''
		response = self.llm_client.call(messages=messages, model=self.model, **kwargs)
		return response

	def rag_llm_call(self, query: str, k: int = 3, **kwargs):
		'''
		Allows the user to conduct a RAG assisted LLM call.
		Returns a RAGLLMResponse object consisting of:
		- success: boolean
		- response: response object from the API
		'''
		response = self.rag_llm_client.call(query=query, k=k, **kwargs)
		return response

	def process_new_documents(self,
		loader_kwargs: dict = None,
		splitter_chunk_size: int = None,
		splitter_chunk_overlap: int = None,
		splitter_chunk_len_function: Callable = None,
		separators: list = None,
		splitter_kwargs: dict = None
	):
		'''
		Allows the user to trigger processing of new .PDF documents for RAG assisted LLM calls.
		'''

		self.rag_document_processor.process_new_documents(
			loader_kwargs=loader_kwargs or {},
			splitter_chunk_size=splitter_chunk_size,
			splitter_chunk_overlap=splitter_chunk_overlap,
			splitter_chunk_len_function=splitter_chunk_len_function,
			separators=separators,
			splitter_kwargs=splitter_kwargs or {}
		)