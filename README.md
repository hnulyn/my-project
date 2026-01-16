# DebateCV  

DebateCV is a claim verifictaion system centered on multi-round debates between two agents (affirmative and negative) moderated by a judge-style assistant.  

## Key components

- `inference.py`: LoRA-based zero-shot inference for moderator verdicts over `dev` outputs.
- `scripts/run_debate.py`: Main runner for orchestrating debates with configurable models and prompt templates.
- `config/default.json`: Default configuration for models, prompts, and API endpoints (API keys cleared).
- `src/`: Core implementation (agents, utilities, data models) used by the inference scripts.
- `output/`: Stores inference results produced by `scripts/run_debate.py`.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the data:
   - **Claims data**: Download the AVeriTeC dataset from [Hugging Face](https://huggingface.co/chenxwh/AVeriTeC) and place the `train.json` and `dev.json` files in `./data/AVeriTeC/` (or set `DATA_BASE_DIR` environment variable to your data directory).
   - **Evidence data**: Download the HerO evidence files from the [HerO repository](https://github.com/ssu-humane/HerO/tree/main/data_store/baseline) and place them in `./data/HerO/data_store/baseline/`:
     - `dev_veracity_prediction_8b.json`
   
   The expected directory structure:
   ```
   ./data/
   ├── AVeriTeC/
   │   ├── train.json
   │   └── dev.json
   └── HerO/
       └── data_store/
           └── baseline/
               └── dev_veracity_prediction_8b.json
   ```

3. Provide API credentials:
   - Set `OPENAI_API_KEY`, `LLAMA_API_KEY`, or `QWEN_API_KEY` in your environment, or edit `config/default.json` with non-empty URLs and keys before running.

4. Run a zero-shot debate:
   ```bash
   python scripts/run_debate.py \
     --config config/default.json \
     --dataset dev \
     --evidence hero \
     --workers 1
   ```

5. Process moderator verdicts from completed debates:
   ```bash
   python inference.py --dev_dir output/dev/<run_folder>
   ```

## Configuration

The `config/default.json` file controls:
- `models`: debate and judge models plus generation settings.
- `api_settings`: API endpoints and sleep intervals.
- `prompts`: Prompt templates for each role.
- `debate_settings`: Debate length and verbosity.

Adjust these entries to match your inference targets.

## Outputs

- `scripts/run_debate.py` writes debate results and metadata into `output/` (one JSON file per claim).
- `inference.py` reads from a `dev` directory and appends moderator verdicts to each file, producing a `_Ilocal` directory.

* Trained model checkpoints and the dataset will be uploaded to Hugging Face upon acceptance.

## 🙏 Acknowledgments

- This project uses the AVeriTeC dataset, available at [https://huggingface.co/chenxwh/AVeriTeC](https://huggingface.co/chenxwh/AVeriTeC). We thank the creators for making it publicly accessible.