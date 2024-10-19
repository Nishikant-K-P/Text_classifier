# Text Classification for Property-related Cases

This project classifies property-related text data to detect illegal activities, such as break-ins, squatter or trespasser activity, property damage, or theft. It uses **Hugging Face Transformers** models for summarization and classification.

## Features

- **Data Cleaning:** Cleans text by removing irrelevant information such as URLs, emails, and phone numbers.
- **Text Summarization:** Summarizes case descriptions using the **BART model**.
- **Text Classification:** Classifies text for illegal activities using the **T5 model**.
- **Output:** Exports predictions to an Excel file for analysis.

## Tech Stack

- **Python** (v3.8+)
- **Hugging Face Transformers**: T5, BART models
- **Pandas**: For data manipulation
- **PyTorch**: Model inference
- **re**: For text cleaning
- **tqdm**: For progress tracking

## Prerequisites

- Python 3.8+
- CUDA-enabled GPU for faster processing (optional but recommended)
- Install dependencies:
  ```bash
  pip install transformers torch pandas tqdm
