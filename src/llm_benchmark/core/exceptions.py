

class LLMBenchmarkError(Exception):
    
    pass

class ConfigurationError(LLMBenchmarkError):
    
    pass

class ProviderError(LLMBenchmarkError):
    
    pass

class APIError(ProviderError):
    
    pass

class RateLimitError(APIError):
    
    pass

class AuthenticationError(APIError):
    
    pass

class TaskError(LLMBenchmarkError):
    
    pass

class DatasetError(LLMBenchmarkError):
    
    pass

class EvaluationError(LLMBenchmarkError):
    
    pass

class MetricError(EvaluationError):
    
    pass
