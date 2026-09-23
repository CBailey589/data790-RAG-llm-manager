from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class LLMResponse:
    success: bool
    response: Any

class LlmClient:
	'''
	Tracks API calls to the configured LLM.
	'''
	def __init__(
		self,
		llm_base_url: str,
		llm_api_key: str
	):
		'''
		Creates an instance of the LlmClient class.
		'''
		self.client = OpenAI(base_url = llm_base_url, api_key = llm_api_key)

	def call(
		self,
		messages: list,
		model: str,
		**kwargs
	) -> dict:
		'''
		Performs a call to the LLM endpoint configured in .env.
		INPUTS:
		- messages: The prompt to send to the model
		- model: The desired model (ex: "gpt-4.1-mini"). Defaults to .env configured model if no model is provided.
		- kwargs: model specific inputs

		RETURNS:
		- Success bool
		- LLM response object / returned error if unsuccessful
		'''

		model = model or self.default_model

		try:
			response = self.client.chat.completions.create(
				model=model,
				messages=messages,
				**kwargs
			)

			return LLMResponse(success=True, response=response)

		except Exception as e:

			return LLMResponse(success=False, response=e)