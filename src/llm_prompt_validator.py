class LLMPromptValidator:
	'''
	Performs basic security and input validation for LLM requests.
	'''

	def __init__(self, max_input_length: int = 10000):
		self.max_input_length = max_input_length
		self.suspicious_patterns = [
			'ignore previous instructions',
			'ignore all instructions',
			'disregard your instructions',
			'print your system prompt',
			'system prompt',
			'<assistant>',
			'assistant response:',
			'<<<end>>>',
			'drop table',
			'delete from',
			'authorized security audit',
			'i am a developer testing your system',
		]

	def validate_prompt(self, prompt: str) -> tuple[bool, str]:
		'''
		Validates user prompt before sending it to an LLM.

		Returns:
		- Success bool
		- 'Valid Prompt' / reason for failed security check
		'''
		if prompt is None or len(prompt.strip()) == 0:
			return False, 'LLM prompt is empty.'

		if len(prompt) > self.max_input_length:
			return False, f'Prompt exceeds maximum length ({self.max_input_length})'

		lower_input = prompt.lower()
		for pattern in self.suspicious_patterns:
			if pattern in lower_input:
				return False, f"Suspicious pattern detected: '{pattern}'"

		return True, 'Valid Prompt'