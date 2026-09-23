import os
from typing import Callable

from dotenv import load_dotenv

from .llm_client import LlmClient
from .rag_document_processor import RagDocumentProcessor


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
		embedding_model: str = None
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
		self.default_model = default_model
		self.default_embedding_model = default_embeddings_model

		self.llm_client = LlmClient()

		embedding_model = embedding_model or self.default_embedding_model
		self.rag_document_processor = RagDocumentProcessor(embedding_model)

	def basic_llm_call(
		self,
		messages: list,
		model: str = None,
		**kwargs
	):
		model = model or self.default_model

		response = self.llm_client.call(messages=messages, model=model, **kwargs)

		if response.success:
			print(f"successful call {response.response}")
		else:
			print(f"failed call {response.response}")

	def process_new_documents(
		self,
		loader_kwargs: dict = {},
		splitter_chunk_size: int = None,
		splitter_chunk_overlap: int = None,
		splitter_chunk_len_function: Callable = None,
		separators: list = None,
		splitter_kwargs: dict = {}
	):
		self.rag_document_processor.process_new_documents(
			loader_kwargs=loader_kwargs,
			splitter_chunk_size=splitter_chunk_size,
			splitter_chunk_overlap=splitter_chunk_overlap,
			splitter_chunk_len_function=splitter_chunk_len_function,
			separators=separators,
			splitter_kwargs=splitter_kwargs
		)




