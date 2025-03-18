---
title: Mehfil-e-Sukhan
emoji: 📜
colorFrom: "red"
colorTo: "gray"
sdk: streamlit
sdk_version: "1.43.0"
app_file: app.py
pinned: false
---

# Mehfil-e-Sukhan: Har Lafz Ek Mehfil

An AI-powered Roman Urdu poetry generation application using BiLSTM neural networks.

## Overview

Mehfil-e-Sukhan ("Poetry Gathering" in Urdu) is an interactive application that generates Roman Urdu poetry based on a starting word or phrase provided by the user. The application uses a Bidirectional LSTM neural network trained on a curated dataset of Roman Urdu poetry.

## Features

- **Custom Poetry Generation**: Generate Roman Urdu poetry from any starting word or phrase.
- **Adjustable Parameters**:
  - **Number of Words**: Control the length of generated poetry (12-48 words).
  - **Creativity (Temperature)**: Adjust the randomness in word selection (0.5-2.0).
  - **Focus (Top-p)**: Fine-tune how closely the model adheres to probable word sequences (0.5-1.0).
- **Elegant Interface**: Dark-themed UI designed specifically for poetry presentation.
- **Automatic Formatting**: Output is automatically formatted into poetic lines.

## How to Use

1. Enter a starting word or phrase in Roman Urdu (e.g., "ishq", "zindagi", "mohabbat").
2. Adjust the generation parameters:
   - Number of Words: Select how many words you want in your poem.
   - Creativity: Higher values (>1.0) produce more unique but potentially less coherent poetry. Lower values (<1.0) create more predictable output.
   - Focus: Higher values make the AI stick to more probable word combinations.
3. Click "Generate Poetry" and wait for your custom poem to appear.

## Technical Details

- **Model**: Bidirectional LSTM with 3 layers
- **Tokenization**: SentencePiece with BPE encoding
- **Vocabulary Size**: 12,000 tokens
- **Text Generation**: Nucleus (top-p) sampling for balanced creativity and coherence

## Installation for Local Development

If you want to run the application locally:

```bash
# Clone the repository
git clone https://github.com/yourusername/Mehfil-e-Sukhan.git
cd Mehfil-e-Sukhan

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Requirements

- Python 3.8+
- torch==2.6.0
- sentencepiece==0.2.0
- huggingface-hub==0.29.3
- streamlit==1.43.0

## Project Structure

```
Mehfil-e-Sukhan/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
└── README.md           # This documentation
```

The model weights and SentencePiece model are stored on Hugging Face Hub and are downloaded automatically when the application runs.

## How It Works

1. **Data Processing**: The model was trained on a curated dataset of Roman Urdu poetry lines.
2. **Tokenization**: Text was tokenized using SentencePiece's BPE algorithm.
3. **Model Training**: A Bidirectional LSTM architecture was trained to predict the next token in a sequence.
4. **Text Generation**: At inference time, nucleus sampling is used to select the next word with a balance of creativity and coherence.
5. **Formatting**: Generated text is automatically formatted into lines with alternating indentation for aesthetic presentation.

## Model and Dataset

- **Model**: You can find the complete model, weights, and training notebooks on Hugging Face:
  [Mehfil-e-Sukhan on Hugging Face](https://huggingface.co/zaiffi/Mehfil-e-Sukhan)
- **Dataset**: The model was trained on the Roman Urdu Poetry dataset available on Kaggle:
  [Roman Urdu Poetry Dataset](https://www.kaggle.com/datasets/mianahmadhasan/roman-urdu-poetry-csv)

## Limitations

- The current model was trained on a relatively small dataset (~1300 lines), which may occasionally result in repetitive patterns.
- Roman Urdu is not standardized, so the model may struggle with unusual spellings or transliterations.
- Generation speed depends on available computational resources.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

- LinkedIn: [Muhammad Huzaifa Saqib](https://www.linkedin.com/in/muhammad-huzaifa-saqib-90a1a9324/)
- GitHub: [zaiffishiekh01](https://github.com/zaiffishiekh01)
- Email: [zaiffishiekh@gmail.com](mailto:zaiffishiekh@gmail.com)

## Acknowledgements

- Poetry is the rhythmical creation of beauty in words - Edgar Allan Poe
