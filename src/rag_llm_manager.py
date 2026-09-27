import os
from typing import Callable

from dotenv import load_dotenv

from .budget_manager import BudgetManager
from .cost_tracker import CostTracker
from .llm_client import LLMClient
from .prompt_cache import PromptCache
from .rag_document_processor import RAGDocumentProcessor
from .rag_llm_client import RAGLLMClient


class RAGLLMManager:
	'''
	Main Class for managing LLM API interaction. This class provides:
	- A Basic LLM API client used to make one off calls to a configured model.
	- A RAG LLM API client used to make RAG assisted calls to a configured model.
	- A Cost Tracker that tracks API calls of various types and calculated costs associated with them based on pricing information.
	- A Budget Manager that allows users to set a maximum budget and track their spending against that budget.
	- A Prompt Cache that will return deterministic results for identical LLM API queries, reducing LLM API costs.
	- A Document Processor that allows users to upload documents and process them for RAG assisted LLM API calls.
	'''

	def __init__(
		self,
		model: str = None,
		embeddings_model: str = None,
		daily_budget: float = 1.00,
		enable_budget_hard_cap: bool = True,
		max_cache_size: int = 100
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
		self.rag_temperature  = 0

		self.llm_client = LLMClient()

		embeddings_model = embeddings_model or self.default_embeddings_model
		self.rag_document_processor = RAGDocumentProcessor(embeddings_model)
		self.rag_llm_client = RAGLLMClient(embedding_model=embeddings_model, model=self.model)

		self.cost_tracker = CostTracker(model=self.model)

		if daily_budget <= 0:
			raise ValueError("[$$$] DAILY BUDGET MUST BE GREATER THAN ZERO.")
		self.budget_manager = BudgetManager(daily_budget=daily_budget, enable_budget_hard_cap=enable_budget_hard_cap)

		self.prompt_cache = PromptCache(max_size=max_cache_size)

	def set_llm_model(self, model: str, **kwargs):
		'''
		Allows the user to change the LLM model (ex: 'gpt-4.1-mini', 'gpt-5.6-sol')
		'''
		# Send to Cost Tracker first so we can update Pricing DB if necessary, will raise error if pricing info is unknown
		self.cost_tracker.update_model(model=model)
		self.model = model
		self.rag_temperature = kwargs.get('temperature', 0)
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

		# First check with prompt cache as returning cached responses is free:
		temperature = kwargs.get('temperature', 0)
		cached_response = self.prompt_cache.get_cached_response(request=messages, model=self.model, temperature=temperature, call_type='basic')

		if cached_response is not None:
			self.cost_tracker.record_call(response=cached_response, latency_ms=0, prompt_tokens=0, completion_tokens=0, total_tokens=0, cache_hit=True)
			return cached_response

		# No cached response, attempt LLM call
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
				self.cost_tracker.record_call(
					response=basic_llm_response,
					latency_ms=basic_llm_response.latency,
					prompt_tokens=prompt_tokens,
					completion_tokens=completion_tokens,
					total_tokens=total_tokens,
					cache_hit=False
				)
				self.prompt_cache.set_cached_response(
					request=messages,
					model=self.model,
					temperature=temperature,
					call_type='basic',
					response=basic_llm_response
				)

			else:
				self.cost_tracker.record_call(
					response=basic_llm_response,
					latency_ms=basic_llm_response.latency,
					prompt_tokens=0,
					completion_tokens=0,
					total_tokens=0,
					cache_hit=False,
					error=basic_llm_response.response
				)

			return basic_llm_response

	def rag_llm_call(self, query: str, k: int = 3, **kwargs):
		'''
		Allows the user to conduct a RAG assisted LLM call.
		Returns a RAGLLMResponse object consisting of:
		- success: boolean
		- response: response object from the API
		'''

		# First check with prompt cache as returning cached responses is free:
		rag_request = {'query': query, 'k': k}
		cached_response = self.prompt_cache.get_cached_response(request=rag_request, model=self.model, temperature=self.rag_temperature, call_type='rag')

		if cached_response is not None:
			self.cost_tracker.record_call(response=cached_response, latency_ms=0, prompt_tokens=0, completion_tokens=0, total_tokens=0, cache_hit=True)
			return cached_response

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
				self.cost_tracker.record_call(
					response=rag_llm_response,
					latency_ms=rag_llm_response.latency,
					prompt_tokens=prompt_tokens,
					completion_tokens=completion_tokens,
					total_tokens=total_tokens,
					cache_hit=False
				)
				self.prompt_cache.set_cached_response(
					request=rag_request,
					model=self.model,
					temperature=self.rag_temperature,
					call_type='rag',
					response=rag_llm_response
				)

			else:
				self.cost_tracker.record_call(
					response=rag_llm_response,
					latency_ms=rag_llm_response.latency,
					prompt_tokens=0,
					completion_tokens=0,
					total_tokens=0,
					cache_hit=False,
					error=rag_llm_response.response
				)

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

	def update_cache_max_size(self, max_size: int):
		'''
		Updates the maximum number of LLM API responses that can be held in message cache.
		'''
		self.prompt_cache.update_cache_max_size(max_size=max_size)