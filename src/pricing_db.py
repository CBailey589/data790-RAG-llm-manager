PRICING_DB = {
    # OpenAI / Azure OpenAI
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016, "provider": "OpenAI/Azure"},
    "gpt-4o": {"input": 0.0025, "output": 0.01, "provider": "OpenAI/Azure"},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "provider": "OpenAI/Azure"},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03, "provider": "OpenAI/Azure"},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015, "provider": "OpenAI/Azure"},
    "gpt-5.6-sol": {"input": 0.004, "output": 0.02, "provider": "OpenAI/Azure"},

    # Anthropic
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015, "provider": "Anthropic"},
    "claude-3-opus": {"input": 0.015, "output": 0.075, "provider": "Anthropic"},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125, "provider": "Anthropic"},

    # Google
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005, "provider": "Google"},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003, "provider": "Google"},

    # Embeddings
    "text-embedding-ada-002": {"input": 0.0001, "output": 0, "provider": "OpenAI/Azure"},
    "text-embedding-3-small": {"input": 0.00002, "output": 0, "provider": "OpenAI"},
}