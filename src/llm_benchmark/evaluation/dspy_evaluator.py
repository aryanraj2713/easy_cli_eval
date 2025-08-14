from typing import Any, Dict, List, Optional, Union, Callable

import dspy
from dspy.evaluate import Evaluate

from ..core.exceptions import ConfigurationError, EvaluationError
from ..providers.base import BaseLLMProvider
from .base import BaseEvaluator
from .metrics.dspy_metrics import create_dspy_metric, evaluate_with_dspy

class DSPyEvaluator(BaseEvaluator):
    
    def __init__(self, provider: BaseLLMProvider, **kwargs):
        super().__init__(provider, **kwargs)
        
        self.method = kwargs.get("method", "chain_of_thought")
        self.num_threads = kwargs.get("num_threads", 1)
        self.metric_name = kwargs.get("metric", "accuracy")
        
        try:
            self.dspy_lm = provider.to_dspy()
            dspy.configure(lm=self.dspy_lm)
        except Exception as e:
            raise ConfigurationError(f"Failed to configure DSPy: {str(e)}")
    
    def evaluate(self, model: BaseLLMProvider, dataset: str, config: Dict[str, Any]) -> Dict[str, float]:
        try:
            program = self._create_program(config.get("task", ""))
            
            examples = self._load_dataset(dataset, config.get("num_samples", 10))
            
            dspy_examples = [
                dspy.Example(
                    input=example.get("input", example.get("question", "")),
                    output=example.get("output", example.get("answer", ""))
                ) for example in examples
            ]
            
            metric_name = self._get_metric_name(config.get("task", ""))
            
            eval_results = evaluate_with_dspy(
                program=program,
                dataset=dspy_examples,
                metric_name=metric_name,
                num_threads=self.num_threads
            )
            
            return {
                metric_name: eval_results["score"],
                "detailed_results": eval_results["results"]
            }
            
        except Exception as e:
            raise EvaluationError(f"DSPy evaluation failed: {str(e)}")
    
    def get_required_config(self) -> List[str]:
        return ["task"]
    
    def _create_program(self, task: str) -> dspy.Module:
        if self.method == "chain_of_thought":
            return self._create_chain_of_thought(task)
        elif self.method == "few_shot":
            return self._create_few_shot(task)
        else:
            return self._create_zero_shot(task)
    
    def _create_chain_of_thought(self, task: str) -> dspy.Module:
        class ChainOfThoughtProgram(dspy.Module):
            def __init__(self, task_description):
                super().__init__()
                self.task_description = task_description
                self.predictor = dspy.ChainOfThought("input -> output")
            
            def forward(self, input):
                return self.predictor(input=input)
        
        return ChainOfThoughtProgram(task)
    
    def _create_few_shot(self, task: str) -> dspy.Module:
        examples = self.kwargs.get("examples", [])
        
        if not examples:
            raise ConfigurationError("Few-shot evaluation requires examples")
        
        class FewShotProgram(dspy.Module):
            def __init__(self, task_description, examples):
                super().__init__()
                self.task_description = task_description
                self.examples = examples
                self.predictor = dspy.FewShotExample(examples, "input -> output")
            
            def forward(self, input):
                return self.predictor(input=input)
        
        return FewShotProgram(task, examples)
    
    def _create_zero_shot(self, task: str) -> dspy.Module:
        class ZeroShotProgram(dspy.Module):
            def __init__(self, task_description):
                super().__init__()
                self.task_description = task_description
                self.predictor = dspy.Predict("input -> output")
            
            def forward(self, input):
                return self.predictor(input=input)
        
        return ZeroShotProgram(task)
    
    def _get_metric_name(self, task: str) -> str:
        if task == "question_answering" or task == "qa":
            return "contains"
        elif task == "summarization":
            return "semantic_similarity"
        else:
            return self.metric_name
    
    def _load_dataset(self, dataset: str, num_samples: int) -> List[Dict[str, Any]]:
        from ...benchmarks.datasets.loader import load_dataset
        return load_dataset(dataset, num_samples)
