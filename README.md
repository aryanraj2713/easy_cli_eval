# LLM Benchmark CLI

Production-ready Python CLI tool for benchmarking Large Language Models with modular architecture and comprehensive evaluation capabilities. Compare performance across providers (OpenAI, Gemini, Grok/X.AI) using traditional methods and GEPA (Genetic Prompt Architecture).

## Features

- **Multiple LLM Providers**: Support for OpenAI (GPT-5, GPT-4o), Google Gemini (Gemini 2.5 Pro), and Grok/X.AI (Grok-2)
- **Modular Architecture**: Easily extend with new providers, evaluation methods, and metrics
- **GEPA Integration**: Genetic prompt optimization using DSPy's GEPA library with evaluation support
- **Traditional Methods**: Zero-shot, few-shot, and chain-of-thought prompting
- **Comprehensive Metrics**: Accuracy, F1, ROUGE, BLEU, and more
- **Experiment Configuration**: YAML-based experiment definitions
- **Structured Logging**: Detailed logs with configurable levels

## Quickstart

1) Install UV package manager (if not installed):
   - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

2) Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3) Install in editable mode with desired extras:
   ```bash
   uv pip install -e .[openai,gemini,metrics]
   ```

4) Set up environment variables (create a `.env` file or export directly):
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   GROK_API_KEY=your_grok_api_key
   ```

5) Run help to see available commands:
   ```bash
   llm-benchmark --help
   ```

## Available Commands

### Run a Single Benchmark

```bash
llm-benchmark run --provider openai --model gpt-5 --task qa --method zero_shot
```

Options:
- `--provider`: Provider name (openai, gemini, grok)
- `--model`: Model name (e.g., gpt-5, gemini-2.5-pro)
- `--task`: Task name (qa, summarization, etc.)
- `--method`: Evaluation method (zero_shot, few_shot, chain_of_thought, gepa, dspy)
- `--dataset`: Dataset name (optional, defaults to standard dataset for task)
- `--num-samples`: Number of samples to evaluate (default: 10)
- `--output`: Output file path (optional)

### Compare Multiple Models

```bash
llm-benchmark compare --models openai:gpt-5,gemini:gemini-2.5-pro --task summarization
```

Options:
- `--models`: Comma-separated list of models to compare (format: provider:model)
- `--task`: Task name (qa, summarization, etc.)
- `--method`: Evaluation method (default: zero_shot)
- `--dataset`: Dataset name (optional)
- `--num-samples`: Number of samples to evaluate (default: 10)
- `--output`: Output file path (optional)

### Run an Experiment

```bash
llm-benchmark experiment --config configs/experiments/gepa_vs_traditional.yaml
```

Options:
- `--config`: Path to experiment configuration file
- `--output-dir`: Output directory for results (optional)

### List Available Models

```bash
llm-benchmark list-models --provider openai
```

### Validate Configuration File

```bash
llm-benchmark validate-config --file configs/experiments/gepa_vs_traditional.yaml
```

## Environment Variables

| Variable | Description | Required For |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | OpenAI API key | OpenAI provider |
| `GOOGLE_API_KEY` | Google API key | Gemini provider |
| `GROK_API_KEY` | Grok/X.AI API key | Grok provider |
| `GROK_BASE_URL` | Grok API base URL | Grok provider (optional) |
| `LOG_LEVEL` | Logging level (INFO, DEBUG, etc.) | All (optional) |

## Extending the Project

### Adding a New Provider

1. Create a new file in `src/llm_benchmark/providers/` (e.g., `groq.py`)
2. Implement a class that inherits from `BaseLLMProvider`
3. Register the provider using the `@ProviderFactory.register()` decorator
4. Implement required methods: `generate()`, `batch_generate()`, `get_available_models()`, `to_dspy()`, and `gepa()`
5. Add any provider-specific dependencies to `pyproject.toml` as an optional extra

Example:
```python
from .base import BaseLLMProvider
from .factory import ProviderFactory

@ProviderFactory.register("groq")
class GroqProvider(BaseLLMProvider):
    # Implementation here
```

### Adding a New Evaluation Method

1. Create a new file in `src/llm_benchmark/evaluation/` (e.g., `rag.py`)
2. Implement a class that inherits from `BaseEvaluator`
3. Implement required methods: `evaluate()` and `get_required_config()`
4. Update the CLI commands to support the new method

### Adding a New Metric

1. Add the metric implementation to `src/llm_benchmark/evaluation/metrics/calculator.py`
2. Update the metric calculation in the `calculate_metrics()` function
3. Add any required dependencies to the `metrics` extra in `pyproject.toml`

## Work in Progress

- **Grok Integration**: Support for Grok/X.AI's API is currently in development
- **Advanced RAG Methods**: Retrieval-augmented generation techniques will be added in future releases
- **GUI Dashboard**: A web-based dashboard for visualizing benchmark results is planned

## Notes

- Provider SDKs are installed as optional extras to keep the base installation lightweight
- GEPA implementation uses DSPy's GEPA library for genetic prompt optimization with integrated evaluation
- The project follows a modular design to make it easy to extend with new capabilities
- All commands support the `--log-level` option to control verbosity

## License

MIT