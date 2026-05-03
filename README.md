# CSCI544_NLP_Team9_Code
Code Files for CSCI544 Course Project

# About the Project in brief
This project investigates the relationship between large language model (LLM) scale, prompting strategies, and factual reliability. We conducted a controlled empirical study using the **Pythia suite** across five scales (70M, 160M, 410M, 1B, 2.8B) to isolate the impact of model size on hallucination rates. The evaluation covers four diverse benchmarks:

* **TruthfulQA**: Measures adversarial truthfulness and the model's tendency to mimic human falsehoods.
* **FEVER**: Evaluates evidence-based claim verification (SUPPORTS vs. REFUTES).
* **TriviaQA**: Assesses open-domain factual recall.
* **SQuAD 2.0** Extracts answer spans from a passage and abstains on unanswerable questions.

The project compares base Pythia models against instruction-tuned **Qwen2.5 (1.5B, 3B)** models under three prompting strategies: **Zero-shot**, **Few-shot**, and **Chain-of-Thought (CoT)**. Key findings show that while scaling improves factual recall in open domains, it has limited effect on adversarial truthfulness, whereas instruction tuning provides significant gains that scaling alone does not.

# How to Run
The code for TruthfulQA, FEVER, and TriviaQA is provided in a Jupyter Notebook format (`NLP_Project(Fever,TruthfulQA & Trivia).ipynb`). To run the project:

1.  **Environment Setup**: Use an environment with GPU support (e.g., Google Colab or a local machine with a CUDA-enabled GPU like an NVIDIA RTX PRO 6000).
2.  **Install Dependencies**: Run the installation cells at the beginning of the notebook to install necessary libraries:
    ```bash
    pip install transformers accelerate datasets torch pandas rouge-score bert-score matplotlib
    ```
3.  **Execute Cells**: Run the cells sequentially. The notebook will automatically:
    * Download the required datasets from Hugging Face and external sources.
    * Load the Pythia and Qwen models from the Hugging Face Hub.
    * Perform inference and evaluate the models based on the specified metrics.

# Parts of the Code
The notebook is organized into the following logical sections:

* **Dataset Loading and Preprocessing**: Handles the loading, cleaning, and prompt formatting for TruthfulQA, FEVER, and TriviaQA. It converts raw data into unified prompt templates for consistency across evaluations.
* **Model Loading Functions**: Contains utility functions to load different Pythia and Qwen model sizes in appropriate precision (FP16) for inference.
* **Inference Engine**: Manages the tokenization and text generation process for each prompt across different strategies (Zero-shot, Few-shot, CoT).
* **Evaluation Metrics**: Implements various scoring mechanisms including:
    * **Classification Metrics**: Accuracy, Hallucination Rate, Precision, Recall, and F1 for TruthfulQA and FEVER.
    * **Text Metrics**: Token-level F1, Exact Match (EM), ROUGE, and BERTScore for open-domain answers in TriviaQA.
* **Analysis and Visualization**: Scripts to aggregate results and generate scaling curves to visualize performance changes across model sizes and prompting strategies.

# SQuAD 2.0
We additionally evaluated hallucination behavior on SQuAD 2.0, an extractive QA benchmark that includes both answerable and unanswerable questions. The code for SQuAD 2.0 is provided in a Jupyter Notebook format (`nlp_project_squad.ipynb`). SQuAD 2.0 was only evaluated under zero-shot and few-shot prompting. Chain-of-thought was omitted because SQuAD 2.0 expects short answer spans, which contradicts CoT's long outputs.

The Jupyter Notebook includes all necessary code to run SQuAD 2.0, and is self-contained and can be run end-to-end in Google Colab. The notebook downloads the SQuAD dev set and required scripts (`run_squad.py`, `evaluate-v2.0.py`) before running experiments. Note: the evaluator script expects predictions for the entire SQuAD 2.0 dev set. Afterwards, it will run pythia-70m and Qwen2.5-3B-Instruct on SQuAD 2.0, and run the evaluator script for pythia-70m.

To run a model on SQuAD 2.0:
```bash
python run_squad.py \
  --model_family [pythia OR qwen] \
  --model_name [EleutherAI/pythia-70m OR Qwen/Qwen2.5-3B-Instruct OR any other valid model size] \
  --output_file [name_of_predictions_file.json] \
  --prompt_mode [zero_shot OR few_shot] \
  --num_shots 3 [Use `--num_shots` only for few-shot; omit it for zero-shot]
```

To run the evaluator script:
```bash
python evaluate-v2.0.py dev-v2.0.json [name_of_predictions_file.json]
```