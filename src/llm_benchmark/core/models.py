

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    
    
    provider: str = Field(..., description="Provider name (e.g., 'openai', 'gemini', 'grok')")
    model: str = Field(..., description="Model name (e.g., 'gpt-4', 'gemini-pro')")
    api_key: Optional[str] = Field(None, description="API key (if not using environment variables)")
    api_base: Optional[str] = Field(None, description="API base URL (for custom endpoints)")
    parameters: Dict[str, Any] = Field(default_factory=dict, 
                                      description="Model-specific parameters")

class TaskConfig(BaseModel):
    
    
    name: str = Field(..., description="Task name (e.g., 'question_answering', 'summarization')")
    dataset: str = Field(..., description="Dataset name (e.g., 'squad_v2', 'cnn_dailymail')")
    metrics: List[str] = Field(default_factory=list, 
                              description="Metrics to evaluate (e.g., 'accuracy', 'rouge_l')")
    parameters: Dict[str, Any] = Field(default_factory=dict, 
                                      description="Task-specific parameters")

class TraditionalMethodConfig(BaseModel):
    
    
    methods: List[str] = Field(..., description="Methods to use (e.g., 'zero_shot', 'few_shot')")
    parameters: Dict[str, Any] = Field(default_factory=dict, 
                                      description="Method-specific parameters")

class GapeMethodConfig(BaseModel):
    
    
    population_size: int = Field(10, description="Number of prompt variants per generation")
    generations: int = Field(5, description="Number of evolutionary iterations")
    mutation_rate: float = Field(0.3, description="Probability of prompt mutation")
    fitness_function: str = Field("composite_score", 
                                 description="Function to evaluate prompt fitness")
    parameters: Dict[str, Any] = Field(default_factory=dict, 
                                      description="GAPE-specific parameters")

class MethodsConfig(BaseModel):
    
    
    traditional: Optional[TraditionalMethodConfig] = None
    gape: Optional[GapeMethodConfig] = None

class OutputConfig(BaseModel):
    
    
    format: List[str] = Field(default_factory=lambda: ["json"], 
                             description="Output formats (e.g., 'json', 'csv', 'html')")
    include_plots: bool = Field(False, description="Whether to include plots in the output")
    save_intermediate: bool = Field(False, description="Whether to save intermediate results")
    output_dir: str = Field("./results", description="Directory to save results")

class ExperimentConfig(BaseModel):
    
    
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = None
    models: List[ModelConfig] = Field(..., description="Models to benchmark")
    tasks: List[TaskConfig] = Field(..., description="Tasks to benchmark")
    methods: MethodsConfig = Field(..., description="Methods to use")
    output: OutputConfig = Field(default_factory=OutputConfig, 
                                description="Output configuration")

class BenchmarkResult(BaseModel):
    
    
    model: ModelConfig
    task: TaskConfig
    method: str
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_results: Optional[Dict[str, Any]] = None

class GapeResult(BaseModel):
    
    
    best_prompt: str
    fitness_scores: List[float]
    evolution_history: List[Dict[str, Any]]
    final_metrics: Dict[str, float]
    cost_analysis: Optional[Dict[str, Any]] = None
