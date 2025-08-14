

from typing import Any, Dict, List, Optional

from ..core.exceptions import EvaluationError
from ..providers.base import BaseLLMProvider
from .base import BaseEvaluator

class TraditionalEvaluator(BaseEvaluator):
    
    def __init__(self, provider: BaseLLMProvider, **kwargs):
        super().__init__(provider, **kwargs)
        
        self.method = kwargs.get("method", "zero_shot")
        self.examples = kwargs.get("examples", [])
        self.metrics = kwargs.get("metrics", ["accuracy"])
        
        valid_methods = ["zero_shot", "few_shot", "chain_of_thought"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {self.method}. Must be one of {valid_methods}")
        
        if self.method == "few_shot" and not self.examples:
            raise ValueError("Few-shot prompting requires examples")
    
    def evaluate(self, model: BaseLLMProvider, dataset: str, config: Dict[str, Any]) -> Dict[str, float]:
        
        try:
            samples = self._load_dataset(dataset, config.get("num_samples", 10))
            
            prompts = [self._create_prompt(sample, config.get("task")) for sample in samples]
            responses = model.batch_generate(prompts)
            
            metrics = self._evaluate_responses(samples, responses, config.get("metrics", self.metrics))
            
            return metrics
            
        except Exception as e:
            raise EvaluationError(f"Traditional evaluation failed: {str(e)}")
    
    def get_required_config(self) -> List[str]:
        
        required = ["task"]
        if self.method == "few_shot":
            required.append("examples")
        return required
    
    def _load_dataset(self, dataset: str, num_samples: int) -> List[Dict[str, Any]]:
        
        from ..benchmarks.datasets.loader import load_dataset
        return load_dataset(dataset, num_samples)
    
    def _create_prompt(self, sample: Dict[str, Any], task: str) -> str:
        
        if self.method == "zero_shot":
            return self._create_zero_shot_prompt(sample, task)
        elif self.method == "few_shot":
            return self._create_few_shot_prompt(sample, task)
        elif self.method == "chain_of_thought":
            return self._create_chain_of_thought_prompt(sample, task)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _create_zero_shot_prompt(self, sample: Dict[str, Any], task: str) -> str:
        
        if task == "question_answering":
            return f
            
        elif task == "summarization":
            return f
            
        else:
            return f
    
    def _create_few_shot_prompt(self, sample: Dict[str, Any], task: str) -> str:
        
        if task == "question_answering":
            prompt = "Answer the following questions accurately and concisely:\n\n"
        elif task == "summarization":
            prompt = "Summarize the following texts in a concise and accurate manner:\n\n"
        else:
            prompt = f"Perform the following {task} tasks:\n\n"
        
        for i, example in enumerate(self.examples):
            if task == "question_answering":
                prompt += f"Example {i+1}:\n"
                prompt += f"Question: {example['question']}\n"
                if 'context' in example:
                    prompt += f"Context: {example['context']}\n"
                prompt += f"Answer: {example['answer']}\n\n"
                
            elif task == "summarization":
                prompt += f"Example {i+1}:\n"
                prompt += f"Text: {example['text']}\n"
                prompt += f"Summary: {example['summary']}\n\n"
                
            else:
                prompt += f"Example {i+1}:\n"
                prompt += f"Input: {example['input']}\n"
                prompt += f"Output: {example['output']}\n\n"
        
        if task == "question_answering":
            prompt += "Now answer this question:\n"
            prompt += f"Question: {sample['question']}\n"
            if 'context' in sample:
                prompt += f"Context: {sample['context']}\n"
            prompt += "Answer:"
                
        elif task == "summarization":
            prompt += "Now summarize this text:\n"
            prompt += f"Text: {sample['text']}\n"
            prompt += "Summary:"
                
        else:
            prompt += "Now perform this task:\n"
            prompt += f"Input: {sample['input']}\n"
            prompt += "Output:"
        
        return prompt
    
    def _create_chain_of_thought_prompt(self, sample: Dict[str, Any], task: str) -> str:
        
        if task == "question_answering":
            return f
            
        elif task == "summarization":
            return f
            
        else:
            return f
    
    def _evaluate_responses(
        self, 
        samples: List[Dict[str, Any]], 
        responses: List[str], 
        metrics: List[str]
    ) -> Dict[str, float]:
        
        from ..evaluation.metrics.calculator import calculate_metrics
        
        ground_truth = []
        for sample in samples:
            if "answer" in sample:
                ground_truth.append(sample["answer"])
            elif "summary" in sample:
                ground_truth.append(sample["summary"])
            elif "output" in sample:
                ground_truth.append(sample["output"])
            else:
                ground_truth.append("")
        
        return calculate_metrics(ground_truth, responses, metrics)
