import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .llm_prompt_validator import LLMPromptValidator


@dataclass
class BasicLLMResponse:
	success: bool
	response: Any
	latency: float = 0

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
		self.prompt_validator = LLMPromptValidator()

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
		for message_obj in messages:
			for key in message_obj:
				valid, reason = self.prompt_validator.validate_prompt(message_obj[key])
				if valid == False:
					return BasicLLMResponse(success=False, response=reason)

		call_start_time = time.time()
		try:

			response = self.client.chat.completions.create(model=model, messages=messages, **kwargs)
			call_end_time = time.time()
			call_latency = (call_end_time - call_start_time) * 1000

			return BasicLLMResponse(success=True, response=response, latency=call_latency)

		except Exception as e:
			call_end_time = time.time()
			call_latency = (call_end_time - call_start_time) * 1000
			return BasicLLMResponse(success=False, response=e, latency=call_latency)