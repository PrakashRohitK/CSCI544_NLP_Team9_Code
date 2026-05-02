# CSCI544_NLP_Team9_Code
Code Files for CSCI544 Course Project

# About the Project in brief
This project investigates the relationship between large language model (LLM) scale, prompting strategies, and factual reliability. We conducted a controlled empirical study using the **Pythia suite** across five scales (70M, 160M, 410M, 1B, 2.8B) to isolate the impact of model size on hallucination rates. The evaluation covers three diverse benchmarks:

* **TruthfulQA**: Measures adversarial truthfulness and the model's tendency to mimic human falsehoods.
* **FEVER**: Evaluates evidence-based claim verification (SUPPORTS vs. REFUTES).
* **TriviaQA**: Assesses open-domain factual recall.

The project compares base Pythia models against instruction-tuned **Qwen2.5 (1.5B, 3B)** models under three prompting strategies: **Zero-shot**, **Few-shot**, and **Chain-of-Thought (CoT)**. Key findings show that while scaling improves factual recall in open domains, it has limited effect on adversarial truthfulness, whereas instruction tuning provides significant gains that scaling alone does not.

# How to Run
The code is provided in a Jupyter Notebook format (`NLP_Project(Fever,TruthfulQA & Trivia).ipynb`). To run the project:

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