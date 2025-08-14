

from typing import Any, Dict, List, Optional, Callable, Union

import dspy
from dspy.evaluate import Evaluate

from ..core.exceptions import ConfigurationError, EvaluationError
from ..providers.base import BaseLLMProvider

class GEPAOptimizer:
    
    
    def __init__(
        self,
        provider: BaseLLMProvider,
        base_prompt: str,
        target_task: str,
        population_size: int = 10,
        generations: int = 5,
        mutation_rate: float = 0.3,
        **kwargs
    ):
        
        self.provider = provider
        self.base_prompt = base_prompt
        self.target_task = target_task
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.auto_budget = kwargs.get("auto_budget", "light")
        self.eval_samples = kwargs.get("eval_samples", 3)
        self.dataset = kwargs.get("dataset", None)
        
        try:
            self.dspy_lm = provider.to_dspy()
            # Configure a reflection LM - ideally this would be a strong model
            self.reflection_lm = self.dspy_lm
        except Exception as e:
            raise ConfigurationError(f"Failed to convert provider to DSPy language model: {str(e)}")
    
    def run(self) -> Dict[str, Any]:
        
        try:
            # Configure DSPy to use our language model
            dspy.configure(lm=self.dspy_lm)
            
            # Define the task program
            class TaskProgram(dspy.Module):
                def __init__(self, task_description):
                    super().__init__()
                    self.task_description = task_description
                    self.predictor = dspy.ChainOfThought("input -> output")
                
                def forward(self, input_text):
                    return self.predictor(input=input_text).output
            
            # Create training and validation datasets
            if self.dataset:
                # If dataset is provided, use it
                train_data = self.dataset[:len(self.dataset)//2]
                val_data = self.dataset[len(self.dataset)//2:]
            else:
                # Otherwise create a simple example dataset
                train_data = [
                    dspy.Example(input=f"Sample task input {i}", output=f"Sample output {i}")
                    for i in range(5)
                ]
                val_data = [
                    dspy.Example(input=f"Validation task input {i}", output=f"Validation output {i}")
                    for i in range(3)
                ]
            
            # Define evaluation metric function
            def metric_fn(gold, pred, trace=None):
                # Simple accuracy metric - in a real implementation, this would be more sophisticated
                if hasattr(gold, 'output') and hasattr(pred, 'output'):
                    score = 1.0 if gold.output.strip() == pred.output.strip() else 0.0
                else:
                    score = 0.0
                
                # Return score with feedback for GEPA
                return {
                    'score': score,
                    'feedback': f"The model {'correctly' if score > 0.5 else 'incorrectly'} solved the task. "
                                f"Task description: {self.target_task}. "
                                f"Please improve the reasoning process to better solve this type of problem."
                }
            
            # Create the program
            program = TaskProgram(self.target_task)
            
            # Create the GEPA optimizer
            gepa = dspy.GEPA(
                metric=metric_fn,
                auto=self.auto_budget,
                reflection_lm=self.reflection_lm,
                track_stats=True
            )
            
            # Compile the program with GEPA
            optimized_program = gepa.compile(
                program,
                trainset=train_data,
                valset=val_data
            )
            
            # Evaluate the optimized program
            evaluator = Evaluate(
                optimized_program,
                metric=metric_fn,
                num_threads=1
            )
            
            eval_results = evaluator(val_data)
            
            # Extract the results
            return {
                "best_prompt": str(optimized_program),
                "optimized_program": optimized_program,
                "detailed_results": optimized_program.detailed_results if hasattr(optimized_program, 'detailed_results') else None,
                "eval_results": eval_results,
                "average_score": eval_results.score
            }
            
        except Exception as e:
            raise EvaluationError(f"GEPA optimization failed: {str(e)}")
    
    def evaluate_with_dspy(self, program, dataset):
        """
        Evaluate a DSPy program using dspy.Evaluate.
        
        Args:
            program: The DSPy program to evaluate
            dataset: The dataset to evaluate on
            
        Returns:
            Evaluation results
        """
        try:
            # Define a simple metric function
            def metric_fn(gold, pred):
                if hasattr(gold, 'output') and hasattr(pred, 'output'):
                    return 1.0 if gold.output.strip() == pred.output.strip() else 0.0
                return 0.0
            
            # Create evaluator
            evaluator = Evaluate(
                program,
                metric=metric_fn,
                num_threads=1
            )
            
            # Run evaluation
            results = evaluator(dataset)
            
            return {
                "score": results.score,
                "results": results
            }
        except Exception as e:
            raise EvaluationError(f"DSPy evaluation failed: {str(e)}")
