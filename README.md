# LLM Benchmark CLI

Production-ready Python CLI to benchmark LLMs across providers (OpenAI, Gemini, Grok/X.AI) with modular evaluation methods (traditional, GAPE) and DSPy integration.

## Quickstart

1) Install uv (if not installed):
- macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`

2) Install in editable mode:
- `uv pip install -e .[openai,gemini,metrics]`

3) Run help:
- `llm-benchmark --help`

## Examples

- `llm-benchmark run --provider openai --model gpt-4o --task qa --method gape`
- `llm-benchmark compare --models openai:gpt-4o,gemini:gemini-pro --task summarization`
- `llm-benchmark experiment --config configs/experiments/gape_vs_traditional.yaml`
- `llm-benchmark list-models --provider openai`
- `llm-benchmark validate-config --file configs/experiments/gape_vs_traditional.yaml`

## Environment Variables

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` (Gemini)
- `GROK_API_KEY` (X.AI; OpenAI-compatible endpoint)
- `GROK_BASE_URL` (optional; defaults may change)
- `LOG_LEVEL` (e.g. `INFO`, `DEBUG`)

## Notes

- Provider SDKs are optional extras.
- GAPE is provided as a modular baseline; extend fitness/crossover/mutation as needed.
- Uses DSPy for orchestration hooks; advanced setups can configure DSPy retrievers and evaluators.
