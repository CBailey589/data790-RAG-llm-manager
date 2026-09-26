import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class BasicLLMResponse:
    success: bool
    response: Any

class LLMClient:
	'''
	Makes API calls to the configured LLM.
	'''
	def __init__(self):
		'''
		Creates an instance of the LlmClient class.
		'''
		llm_base_url = os.getenv("LLM_BASE_URL")
		llm_api_key = os.getenv("LLM_API_KEY")
		self.client = OpenAI(base_url = llm_base_url, api_key = llm_api_key)

	def call(self, messages: list, model: str, **kwargs) -> BasicLLMResponse:
		'''
		Performs a call to the LLM endpoint configured in .env.
		INPUTS:
		- messages: The prompt to send to the model
		- model: The desired model (ex: "gpt-4.1-mini"). Defaults to .env configured model or previously set model if no model is provided.
		- kwargs: model specific inputs

		RETURNS BasicLLMResponse class:
		- Success bool
		- LLM response object / returned error if unsuccessful
		'''
		try:
			# TODO: ADD SECURITY CLASS CHECK HERE!!!

			response = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
			return BasicLLMResponse(success=True, response=response)

		except Exception as e:
			return BasicLLMResponse(success=False, response=e)