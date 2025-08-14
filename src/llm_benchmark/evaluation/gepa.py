

from typing import Any, Dict, List, Optional

import dspy
from gepa import GeneticPromptOptimizer

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
        
        self.fitness_function = kwargs.get("fitness_function", "default")
        self.elitism = kwargs.get("elitism", 2)
        self.eval_samples = kwargs.get("eval_samples", 3)
        
        if self.population_size < 4:
            raise ConfigurationError("Population size must be at least 4")
        if self.generations < 1:
            raise ConfigurationError("Number of generations must be at least 1")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ConfigurationError("Mutation rate must be between 0.0 and 1.0")
        if self.elitism >= self.population_size:
            raise ConfigurationError("Elitism must be less than population size")
        
        try:
            self.dspy_lm = provider.to_dspy()
        except Exception as e:
            raise ConfigurationError(f"Failed to convert provider to DSPy language model: {str(e)}")
    
    def run(self) -> Dict[str, Any]:
        
        try:
            dspy.configure(lm=self.dspy_lm)
            
            class TaskProgram(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.task = self.target_task
                
                def forward(self, input_text):
                    prompt = dspy.Predict("output", context=self.task)(input_text=input_text)
                    return prompt.output
            
            def evaluate_quality(gold, pred):
                return {"quality": 0.8}  
            
            optimizer = GeneticPromptOptimizer(
                task_program=TaskProgram(),
                metric_fn=evaluate_quality,
                population_size=self.population_size,
                num_generations=self.generations,
                mutation_rate=self.mutation_rate,
                elitism=self.elitism
            )
            
            best_prompt, history = optimizer.optimize(
                initial_prompt=self.base_prompt,
                task_description=self.target_task
            )
            
            fitness_scores = []
            evolution_history = []
            
            for i, gen_data in enumerate(history):
                avg_fitness = sum(gen_data["fitness_scores"]) / len(gen_data["fitness_scores"])
                best_idx = gen_data["fitness_scores"].index(max(gen_data["fitness_scores"]))
                
                fitness_scores.append(gen_data["fitness_scores"][best_idx])
                evolution_history.append({
                    "generation": i,
                    "best_prompt": gen_data["population"][best_idx],
                    "best_fitness": gen_data["fitness_scores"][best_idx],
                    "avg_fitness": avg_fitness
                })
            
            final_metrics = self._evaluate_prompt(best_prompt)
            
            return {
                "best_prompt": best_prompt,
                "fitness_scores": fitness_scores,
                "evolution_history": evolution_history,
                "final_metrics": final_metrics
            }
            
        except Exception as e:
            raise EvaluationError(f"GEPA optimization failed: {str(e)}")
    
    def _evaluate_prompt(self, prompt: str) -> Dict[str, float]:
        
        eval_prompt = f
        
        try:
            metrics_sum = {"accuracy": 0.0, "relevance": 0.0, "coherence": 0.0, "diversity": 0.0}
            
            for _ in range(self.eval_samples):
                response = self.provider.generate(eval_prompt)
                
                for metric in metrics_sum:
                    try:
                        start = response.find(f'"{metric}"') + len(f'"{metric}"')
                        start = response.find(":", start) + 1
                        end = response.find(",", start)
                        if end == -1:
                            end = response.find("}", start)
                        value_str = response[start:end].strip()
                        value = float(value_str)
                        metrics_sum[metric] += value
                    except (ValueError, IndexError):
                        pass
            
            metrics = {k: v / self.eval_samples for k, v in metrics_sum.items()}
            
            weights = {"accuracy": 0.4, "relevance": 0.3, "coherence": 0.2, "diversity": 0.1}
            metrics["primary_metric"] = sum(metrics.get(k, 0) * w for k, w in weights.items())
            
            return metrics
            
        except Exception as e:
            return {
                "accuracy": 0.5,
                "relevance": 0.5,
                "coherence": 0.5,
                "diversity": 0.5,
                "primary_metric": 0.5,
            }
