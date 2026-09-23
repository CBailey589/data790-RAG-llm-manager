import os

from dotenv import load_dotenv

from .llm_client import LlmClient


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

		self.llm_base_url = llm_base_url
		self.llm_api_key = llm_api_key
		self.defaul_model = default_model

		self.llm_client = LlmClient(llm_base_url = llm_base_url, llm_api_key = llm_api_key)

	def basic_llm_call(
		self,
		messages: list,
		model: str = None,
		**kwargs
	):
		model = model or self.defaul_model

		response = self.llm_client.call(
			messages=messages,
			model=model

		)

		if response.success:
			print(f"successful call {response.response}")
		else:
			print(f"failed call {response.response}")



