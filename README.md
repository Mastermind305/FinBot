fine-tuned Llama 3 (8B) model designed to assist bank employees in Nepal with customer service, policies, and procedures, deployed as a Gradio web interface.

## Overview

This project fine-tunes the [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) model using LoRA (Low-Rank Adaptation) on a dataset of 169 banking-related question-answer pairs (`Dataset.csv`). The resulting model powers **Finbot**, a chatbot deployed via a Gradio interface (`deploy.py`) to help bank employees handle queries about customer disputes, account openings, mobile banking security, and more. The project includes a Jupyter notebook (`finbot.ipynb`) for fine-tuning, evaluation, and ROUGE scoring.

## Files

- **`deploy.py`**: Python script to deploy Finbot as a Gradio-based chat interface, allowing users to interact with the fine-tuned model.
- **`Dataset.csv`**: Dataset containing 169 question-answer pairs on Nepali banking policies, customer service, and employee procedures.
- **`llama3 (2).ipynb`**: Jupyter notebook for fine-tuning the Llama 3 model, evaluating performance, and computing ROUGE scores.
- **`requirements.txt`**: List of Python dependencies required to run the project.
- **`.gitignore`**: Excludes sensitive files (e.g., `.env`, model weights) and unnecessary directories (e.g., `__pycache__`).
- **`.env`**: Stores sensitive environment variables like the Hugging Face token (not committed to the repository).

## Setup Instructions

To set up and run the Finbot Banking Assistant locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/finbot-banking-assistant.git
   cd finbot-banking-assistant
   ```

2. **Install Dependencies**:
   Create a virtual environment (optional) and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add your Hugging Face token:
   ```plaintext
   HF_TOKEN=your_huggingface_token
   ```
   Obtain a token from [Hugging Face](https://huggingface.co/settings/tokens).

4. **Download the Fine-Tuned Model** (Optional):
   The fine-tuned model weights (`merged_model`) are hosted on Hugging Face Hub (e.g., `your-username/finbot-model`). The code in `deploy.py` and `finbot.ipynb` is configured to download them automatically using the Hugging Face token.

5. **Run the Gradio Interface**:
   Launch the Finbot chatbot:
   ```bash
   python deploy.py
   ```
   Access the interface in your browser (typically at `http://localhost:7860`).

6. **Fine-Tune or Evaluate (Optional)**:
   To fine-tune the model or reproduce the evaluation, run `finbot` in Jupyter Notebook or JupyterLab. Ensure you have a GPU and sufficient memory (~16GB VRAM for 4-bit quantization).

## Requirements

See `requirements.txt` for a full list of dependencies. Key packages include:
- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `pandas`
- `tqdm`
- `bitsandbytes`
- `gradio`
- `python-dotenv`
- `evaluate`
- `rouge-score`
- `numpy`
- `torch`

Install them with:
```bash
pip install -r requirements.txt
```

## Model Details

- **Base Model**: Meta-Llama-3-8B
- **Fine-Tuning**:
  - Method: LoRA (rank=16, lora_alpha=64)
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - Dataset: 152 training examples, 17 test examples
  - Training Epochs: 10
  - Learning Rate: 5e-5
  - Quantization: 4-bit (using `bitsandbytes`)
- **Evaluation Metrics**:
  - **Eval Loss**: 0.9683 (indicates decent generalization for a small dataset)

- **Performance Notes**:
  - The model generates coherent responses for banking queries.
  - Suitable for prototyping; production use may require a larger dataset or full fine-tuning.

## Example Usage

**Question**: "What should I tell customers about our mobile banking security features?"
**Generated Response**: "Mobile banking uses advanced encryption technology, multi-factor authentication, and biometric verification to protect customer data. Customers must follow security best practices, such as using strong passwords."

See `finbot` for more sample responses and ROUGE evaluations.

## Security Notes

- The Hugging Face token is stored in `.env` and excluded via `.gitignore` to prevent accidental exposure.
- The repository is recommended to be private to protect sensitive banking data in `Dataset.csv`.
- Large model weights (`merged_model`, `llama3-lora-finetuned`) are excluded from the repository and hosted on Hugging Face Hub.

## Future Improvements

- Expand `Dataset.csv` with more diverse questions and edge cases to improve generalization.
- Increase LoRA rank (e.g., `r=32`) or use full fine-tuning to lower `eval_loss`.
- Add additional evaluation metrics (e.g., BLEU, BERTScore) for a comprehensive assessment.
- Implement automated testing with GitHub Actions.

## License

[MIT License](LICENSE)

## Acknowledgments

- Built with [Hugging Face Transformers](https://huggingface.co/docs/transformers) and [Gradio](https://gradio.app).
- Dataset tailored for Nepali banking context, inspired by real-world banking policies.

---
