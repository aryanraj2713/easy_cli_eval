"""
Traditional prompting methods for LLM Benchmark CLI.

This module contains implementations of traditional prompting methods:
- Zero-shot prompting
- Few-shot prompting
- Chain-of-thought prompting
"""

from typing import Any, Dict, List, Optional

from ..core.exceptions import EvaluationError
from ..providers.base import BaseLLMProvider
from .base import BaseEvaluator


class TraditionalEvaluator(BaseEvaluator):
    """Evaluator for traditional prompting methods."""
    
    def __init__(self, provider: BaseLLMProvider, **kwargs):
        """
        Initialize the traditional evaluator.
        
        Args:
            provider: The LLM provider to use
            **kwargs: Additional evaluator-specific parameters
                - method: The prompting method to use (zero_shot, few_shot, chain_of_thought)
                - examples: List of examples for few-shot prompting
                - metrics: List of metrics to evaluate
        """
        super().__init__(provider, **kwargs)
        
        self.method = kwargs.get("method", "zero_shot")
        self.examples = kwargs.get("examples", [])
        self.metrics = kwargs.get("metrics", ["accuracy"])
        
        # Validate method
        valid_methods = ["zero_shot", "few_shot", "chain_of_thought"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {self.method}. Must be one of {valid_methods}")
        
        # Validate examples for few-shot
        if self.method == "few_shot" and not self.examples:
            raise ValueError("Few-shot prompting requires examples")
    
    def evaluate(self, model: BaseLLMProvider, dataset: str, config: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a model on a dataset using traditional prompting.
        
        Args:
            model: The LLM provider to evaluate
            dataset: The dataset to evaluate on
            config: Evaluation configuration
                - task: The task to evaluate (e.g., 'question_answering', 'summarization')
                - num_samples: Number of samples to evaluate
                - metrics: List of metrics to evaluate
                
        Returns:
            Dict mapping metric names to scores
            
        Raises:
            EvaluationError: If evaluation fails
        """
        try:
            # Load dataset
            samples = self._load_dataset(dataset, config.get("num_samples", 10))
            
            # Generate responses
            prompts = [self._create_prompt(sample, config.get("task")) for sample in samples]
            responses = model.batch_generate(prompts)
            
            # Evaluate responses
            metrics = self._evaluate_responses(samples, responses, config.get("metrics", self.metrics))
            
            return metrics
            
        except Exception as e:
            raise EvaluationError(f"Traditional evaluation failed: {str(e)}")
    
    def get_required_config(self) -> List[str]:
        """
        Return list of required configuration parameters.
        
        Returns:
            List of parameter names required for evaluation
        """
        required = ["task"]
        if self.method == "few_shot":
            required.append("examples")
        return required
    
    def _load_dataset(self, dataset: str, num_samples: int) -> List[Dict[str, Any]]:
        """
        Load samples from a dataset.
        
        Args:
            dataset: The dataset to load
            num_samples: Number of samples to load
            
        Returns:
            List of dataset samples
        """
        # In a real implementation, this would load from a dataset file or API
        # This is a simplified version with mock data
        from ..benchmarks.datasets.loader import load_dataset
        return load_dataset(dataset, num_samples)
    
    def _create_prompt(self, sample: Dict[str, Any], task: str) -> str:
        """
        Create a prompt for a sample based on the prompting method.
        
        Args:
            sample: The dataset sample
            task: The task to evaluate
            
        Returns:
            The prompt to send to the model
        """
        if self.method == "zero_shot":
            return self._create_zero_shot_prompt(sample, task)
        elif self.method == "few_shot":
            return self._create_few_shot_prompt(sample, task)
        elif self.method == "chain_of_thought":
            return self._create_chain_of_thought_prompt(sample, task)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _create_zero_shot_prompt(self, sample: Dict[str, Any], task: str) -> str:
        """
        Create a zero-shot prompt.
        
        Args:
            sample: The dataset sample
            task: The task to evaluate
            
        Returns:
            The zero-shot prompt
        """
        if task == "question_answering":
            return f"""Answer the following question accurately and concisely:

Question: {sample['question']}

Context: {sample.get('context', '')}

Answer:"""
            
        elif task == "summarization":
            return f"""Summarize the following text in a concise and accurate manner:

Text: {sample['text']}

Summary:"""
            
        else:
            # Generic prompt for other tasks
            return f"""Perform the following {task} task:

Input: {sample['input']}

Output:"""
    
    def _create_few_shot_prompt(self, sample: Dict[str, Any], task: str) -> str:
        """
        Create a few-shot prompt with examples.
        
        Args:
            sample: The dataset sample
            task: The task to evaluate
            
        Returns:
            The few-shot prompt
        """
        # Start with task description
        if task == "question_answering":
            prompt = "Answer the following questions accurately and concisely:\n\n"
        elif task == "summarization":
            prompt = "Summarize the following texts in a concise and accurate manner:\n\n"
        else:
            prompt = f"Perform the following {task} tasks:\n\n"
        
        # Add examples
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
        
        # Add the actual question
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
        """
        Create a chain-of-thought prompt.
        
        Args:
            sample: The dataset sample
            task: The task to evaluate
            
        Returns:
            The chain-of-thought prompt
        """
        if task == "question_answering":
            return f"""Answer the following question accurately and concisely.
Think step by step to solve the problem, then provide your final answer.

Question: {sample['question']}

Context: {sample.get('context', '')}

Let's think through this step by step:
1."""
            
        elif task == "summarization":
            return f"""Summarize the following text in a concise and accurate manner.
First, identify the key points, then organize them into a coherent summary.

Text: {sample['text']}

Let's break down the key points:
1."""
            
        else:
            # Generic prompt for other tasks
            return f"""Perform the following {task} task.
Think step by step to solve the problem, then provide your final answer.

Input: {sample['input']}

Let's think through this step by step:
1."""
    
    def _evaluate_responses(
        self, 
        samples: List[Dict[str, Any]], 
        responses: List[str], 
        metrics: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate model responses against ground truth.
        
        Args:
            samples: The dataset samples
            responses: The model responses
            metrics: The metrics to evaluate
            
        Returns:
            Dict mapping metric names to scores
        """
        # In a real implementation, this would use proper metric calculations
        # This is a simplified version with mock evaluations
        from ..evaluation.metrics.calculator import calculate_metrics
        
        # Extract ground truth from samples
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
        
        # Calculate metrics
        return calculate_metrics(ground_truth, responses, metrics)
