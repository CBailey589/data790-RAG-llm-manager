import os

from dotenv import load_dotenv

load_dotenv()

from .budget_manager import BudgetManager
from .cost_tracker import CostTracker
from .rag_llm_client import RAGLLMClient


class RAGLLMManager:
	'''
	Main class for managing RAG LLM interactions. This class provides:
	- A RAG LLM client for retrieval-augmented generation.
	- A Cost Tracker for monitoring API usage and associated costs.
	- A Budget Manager for setting and enforcing API spending limits.

	Assumes an existing, populated ChromaDB collection.
	'''

	def __init__(
		self,
		model: str = None,
		embeddings_model: str = None,
		daily_budget: float = 1.00,
		enable_budget_hard_cap: bool = False
	):

		load_dotenv()
		print("MANAGER 1: Environment loaded", flush=True)

		llm_base_url = os.getenv("LLM_BASE_URL")
		llm_api_key = os.getenv("LLM_API_KEY")
		default_model = os.getenv("DEFAULT_MODEL")
		default_embeddings_model = os.getenv("DEFAULT_EMBEDDINGS_MODEL")

		assert llm_base_url
		assert llm_api_key
		assert default_model
		assert default_embeddings_model

		print("MANAGER 2: Environment validated", flush=True)

		self.model = model or default_model
		self.embeddings_model = embeddings_model or default_embeddings_model

		print("MANAGER 3: Creating RAGLLMClient", flush=True)
		self.rag_llm_client = RAGLLMClient(
			embedding_model=self.embeddings_model,
			model=self.model
		)
		print("MANAGER 4: RAGLLMClient created", flush=True)

		print("MANAGER 5: Creating CostTracker", flush=True)
		self.cost_tracker = CostTracker(model=self.model)
		print("MANAGER 6: CostTracker created", flush=True)

		if daily_budget <= 0:
			raise ValueError("[$$$] DAILY BUDGET MUST BE GREATER THAN ZERO.")

		print("MANAGER 7: Creating BudgetManager", flush=True)
		self.budget_manager = BudgetManager(
			daily_budget=daily_budget,
			enable_budget_hard_cap=enable_budget_hard_cap
		)
		print("MANAGER 8: BudgetManager created", flush=True)
		# '''
		# Initializes RAG LLM Manager.
		# '''
		# load_dotenv()

		# llm_base_url = os.getenv("LLM_BASE_URL")
		# assert llm_base_url, "Missing LLM_BASE_URL — please ensure .env file has the LLM_BASE_URL field populated"
		# llm_api_key = os.getenv("LLM_API_KEY")
		# assert llm_api_key, "Missing LLM_API_KEY — please ensure .env file has the LLM_API_KEY field populated"
		# default_model = os.getenv("DEFAULT_MODEL")
		# assert default_model, "Missing DEFAULT_MODEL — please ensure .env file has the DEFAULT_MODEL field populated"
		# default_embeddings_model = os.getenv("DEFAULT_EMBEDDINGS_MODEL")
		# assert default_embeddings_model, "Missing DEFAULT_EMBEDDINGS_MODEL — please ensure .env file has the DEFAULT_EMBEDDINGS_MODEL field populated"

		# self.model = model or default_model
		# self.embeddings_model = embeddings_model or default_embeddings_model
		# self.rag_llm_client = RAGLLMClient(embedding_model=self.embeddings_model, model=self.model)
		# self.cost_tracker = CostTracker(model=self.model)
		# if daily_budget <= 0:
		# 	raise ValueError("[$$$] DAILY BUDGET MUST BE GREATER THAN ZERO.")
		# self.budget_manager = BudgetManager(daily_budget=daily_budget, enable_budget_hard_cap=enable_budget_hard_cap)

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