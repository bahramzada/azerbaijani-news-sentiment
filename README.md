# Azerbaijani News Sentiment Analysis

A sentiment analysis project for Azerbaijani news headlines using BERT-based transformer models. This project fine-tunes a multilingual BERT model to classify sentiment in Azerbaijani news text as Positive, Negative, or Neutral.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bahramzada/azerbaijani-news-sentiment/blob/main/news-sentiment-model.ipynb)

## Overview

This project implements sentiment analysis for Azerbaijani news headlines using state-of-the-art transformer models. Built with PyTorch and Hugging Face Transformers, the model achieves effective sentiment classification for low-resource Azerbaijani language text. The project includes data preprocessing, model training, and inference capabilities specifically designed for Azerbaijani linguistic characteristics.

## About Azerbaijani Language Processing

Azerbaijani (Azərbaycan dili) is a Turkic language with unique linguistic features that require specialized text processing. This project addresses challenges in Azerbaijani NLP including:
- Unicode normalization for Turkish-specific characters (ı, ğ, ş, ç, ö, ü)
- Azerbaijani stopword filtering
- News domain-specific sentiment patterns
- Limited training data resources for low-resource language processing

## Features

- ✨ **BERT-based Sentiment Classification**: Fine-tuned multilingual BERT model for Azerbaijani text
- ⚡ **Azerbaijani Text Preprocessing**: Specialized preprocessing pipeline for Azerbaijani linguistic features
- 🛠️ **News Headline Scraping**: Automated scraping tools for Azerbaijani news sources
- 📈 **Training Monitoring**: Comprehensive training metrics and evaluation tracking

## Dataset

Details about the dataset used:

- **Source**: [Azerbaijani News Sentiment Dataset](https://www.kaggle.com/datasets/raullte/azerbaijani-news-sentiment-dataset)
- **Size**: 2,364+ labeled samples
- **Language**: Azerbaijani (Azərbaycan dili)
- **Format**: UTF-8 encoded CSV with sentiment labels
- **Classes**: Positive, Negative, Neutral
- **Preprocessing**: Unicode normalization, stopword removal, punctuation cleaning, lowercase conversion

## Model Details

- **Base Model**: bert-base-multilingual-cased
- **Training Epochs**: 15
- **Batch Size**: 8 (train and eval)
- **Max Sequence Length**: 512 tokens
- **Learning Rate**: 2e-5
- **Training Loss**: 0.5783
- **Dropout**: 0.35 (hidden and attention layers)
- **Checkpoints**: Saved every epoch to ./az_sentiment_bert
- **Total Training Steps**: 4,440

## Requirements

List the main dependencies:

```
pandas
scikit-learn
datasets
transformers
torch
numpy
requests
beautifulsoup4
unicodedata
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bahramzada/azerbaijani-news-sentiment.git
cd azerbaijani-news-sentiment
```

2. Install dependencies:
```bash
pip install pandas scikit-learn datasets transformers torch numpy requests beautifulsoup4
```

3. Set up GPU support (optional but recommended):
```bash
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Using Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. Run all cells in the notebook
3. The notebook will:
   - Load and preprocess the Azerbaijani sentiment dataset
   - Fine-tune the multilingual BERT model
   - Evaluate model performance and save checkpoints

### Local Usage

1. Open the main notebook:
```bash
jupyter notebook news-sentiment-model.ipynb
```

2. Follow the step-by-step process in the notebook

### Quick Sentiment Prediction Example

```python
# Example Python code for sentiment inference

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

model_path = "./az_sentiment_bert"
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create sentiment analysis pipeline
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Predict sentiment for Azerbaijani text
text = "Bu gözəl xəbərdir, çox sevinirəm"  # "This is good news, I'm very happy"
result = classifier(text)
print(f"Text: {text}")
print(f"Sentiment: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.4f}")
```

## Training Process

The training workflow follows these steps:

1. **Data Loading**: Load the labeled Azerbaijani sentiment dataset
2. **Text Preprocessing**: Apply Azerbaijani-specific text cleaning and normalization
3. **Tokenization**: Convert text to BERT-compatible tokens with multilingual tokenizer
4. **Model Training**: Fine-tune BERT with sentiment classification head
5. **Evaluation**: Monitor training progress and save best model checkpoints

### Training Configuration

```python
# Training configuration used in the project
training_args = TrainingArguments(
    output_dir="./az_sentiment_bert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=15,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=3,
    seed=42
)
```

## File Structure

```
azerbaijani-news-sentiment/
├── README.md                           # This file
├── news-sentiment-model.ipynb          # Main training notebook
├── scrape-headlines-from-qafqazinfo.py # News scraping script
└── .git/                               # Git repository data
```

## Results

Key achievements of the model:

- Successfully fine-tuned multilingual BERT for Azerbaijani sentiment analysis
- Achieved effective sentiment classification across three classes (Positive, Negative, Neutral)
- Final training loss: 0.5783 after 15 epochs
- Model handles Azerbaijani linguistic features and news domain text effectively

## Contributing

Contributions are welcome! Please submit a Pull Request or open an issue to discuss changes or improvements. Particularly welcome:
- Additional Azerbaijani text preprocessing improvements
- Dataset expansion and annotation
- Model architecture enhancements
- Evaluation metric improvements

## License

This project is open source. Please refer to the repository license for specific terms and conditions.

## Acknowledgments

- **Kaggle Dataset Author**: For providing the labeled Azerbaijani sentiment dataset
- **Hugging Face**: For the transformers library and multilingual BERT model
- **QafqazInfo.az**: For providing news content for headline extraction
- **PyTorch & scikit-learn**: For machine learning framework support

## Citation

If you use this project or its resources, please cite:

```bibtex
@misc{azerbaijani-news-sentiment,
  title={Azerbaijani News Sentiment Analysis},
  author={bahramzada},
  year={2024},
  url={https://github.com/bahramzada/azerbaijani-news-sentiment}
}
```

---

*"Azərbaycan dilində hissi analiz - innovasiya və ənənəvi dil emalının birləşməsi"*
*"Sentiment analysis in Azerbaijani - bridging innovation and traditional language processing"*
