import os
from typing import Callable

from dotenv import load_dotenv

from .budget_manager import BudgetManager
from .cost_tracker import CostTracker
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
		embeddings_model: str = None,
		daily_budget: float = 1.00,
		enable_budget_hard_cap: bool = True
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

		self.cost_tracker = CostTracker(model=self.model)

		if daily_budget <= 0:
			raise ValueError("[$$$] DAILY BUDGET MUST BE GREATER THAN ZERO.")
		self.budget_manager = BudgetManager(daily_budget=daily_budget, enable_budget_hard_cap=enable_budget_hard_cap)

	def set_llm_model(self, model: str, **kwargs):
		'''
		Allows the user to change the LLM model (ex: 'gpt-4.1-mini', 'gpt-5.6-sol')
		'''
		# Send to Cost Tracker first so we can update Pricing DB if necessary, will raise error if pricing info is unknown
		self.cost_tracker.update_model(model=model)
		self.model = model
		self.rag_llm_client.update_llm_model(model=self.model, **kwargs)
		print(f"UPDATED LLM MODEL TO: {model}")

	def insert_model_into_pricing_db(self, model_name: str, input_cost: int, output_cost: int, provider: str):
		'''
		Adds a new model to the pricing database for Cost Tracking FOR THIS SESSION ONLY. For permanent addition add the model
		directly to the pricing_db.py file.
		Inputs:
			model_name: str (ex: 'gpt-4.1-mini', 'gpt-5.6-sol')
			input_cost: int - price per 1k input tokens
			output_cost: int - price per 1k output tokens
			provider: str (ex: 'OpenAI/Azure')
		'''
		self.cost_tracker.insert_model_into_pricing_db(model_name=model_name, input_cost=input_cost, output_cost=output_cost, provider=provider)

	def basic_llm_call(self, messages: list, **kwargs):
		'''
		Allows the user to conduct a basic LLM call.
		Returns a BasicLLMResponse object consisting of:
		- success: boolean
		- response: response object from the API
		'''
		# Check with budget manager for permission:
		if self.budget_manager.allow_llm_api_call():
			basic_llm_response = self.llm_client.call(messages=messages, model=self.model, **kwargs)

			if basic_llm_response.success:
				# calculate cost / update budget info:
				prompt_tokens = basic_llm_response.response.usage.prompt_tokens
				completion_tokens = basic_llm_response.response.usage.completion_tokens
				total_tokens = basic_llm_response.response.usage.total_tokens
				api_call_cost = self.cost_tracker.calculate_cost(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
				self.budget_manager.record_cost(cost=api_call_cost)

			return basic_llm_response

	def rag_llm_call(self, query: str, k: int = 3, **kwargs):
		'''
		Allows the user to conduct a RAG assisted LLM call.
		Returns a RAGLLMResponse object consisting of:
		- success: boolean
		- response: response object from the API
		'''
		# Check with budget manager for permission:
		if self.budget_manager.allow_llm_api_call():
			rag_llm_response = self.rag_llm_client.call(query=query, k=k, **kwargs)

			if rag_llm_response.success:
				# calculate cost / update budget info:
				prompt_tokens = rag_llm_response.response['usage']['prompt_tokens']
				completion_tokens = rag_llm_response.response['usage']['completion_tokens']
				total_tokens = rag_llm_response.response['usage']['total_tokens']
				api_call_cost = self.cost_tracker.calculate_cost(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
				self.budget_manager.record_cost(cost=api_call_cost)

			return rag_llm_response

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

	def enable_budget_hard_cap(self):
		'''
		Turns on budget hard cap that prevents LLM API calls if daily budget is exceeded.
		'''
		self.budget_manager.enable_budget_hard_cap()

	def disable_budget_hard_cap(self):
		'''
		Turns off budget hard cap that prevents LLM API calls if daily budget is exceeded.
		'''
		self.budget_manager.disable_budget_hard_cap()

	def update_daily_budget_amount(self, updated_budget_amount:float):
		'''
		Allows a user to update their daily budget for LLM API calls.
		'''
		self.budget_manager.update_daily_budget(daily_budget=updated_budget_amount)