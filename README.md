# News Quality Scoring: An Ensemble of NLP Methods to Assess News Quality

## Authors
- Oday Najad ([oday.najad@studenti.unipd.it](mailto:oday.najad@studenti.unipd.it))
- Tommaso Di Fant ([tommaso.difant@studenti.unipd.it](mailto:tommaso.difant@studenti.unipd.it))
- Wageesha Widuranga ([wageeshawiduranga.waththeliyanage@studenti.unipd.it](mailto:wageeshawiduranga.waththeliyanage@studenti.unipd.it))

## Overview
This project aims to assess various aspects of news article quality using Natural Language Processing (NLP) techniques. The framework evaluates articles based on multiple criteria, including:

- Clickbait Detection
- Information Density Scoring
- Writing Quality Scoring
- AI-Generated Content Detection
- Plagiarism Detection
- Fact-Checking

The goal is to provide an automated system to rate news articles objectively, reducing the influence of misleading or low-quality content in digital media.

## Implementation Details

### 1. Clickbait Detection
**Model:** DistilBERT (DistilBertForSequenceClassification)
- **Dataset:** CSV file with clickbait headlines
- **Training:** Tokenization with DistilBertTokenizer
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score


### 2. Information Density
**Model:** BART (facebook/bart-large-cnn)
- **Dataset:** CNN/DailyMail news dataset
- **Evaluation Metrics:** ROUGE-1, ROUGE-2, ROUGE-L


### 3. Writing Quality Scoring
**Model:** DistilBERT with Multi-Task Learning (MDMT)
- **Dataset:** Automatic Essay Scoring 2.0, CLEAR, ELLIPSE
- **Evaluation Metrics:** Weighted Mean Squared Error (MSE)


### 4. AI-Generated Content Detection
**Model:** DistilBERT with Binary Classification
- **Dataset:** LLM-Detect AI Generated Text, artem9k/ai-text-detection-pile
- **Evaluation Metrics:** Binary Cross Entropy Loss (BCE)


### 5. Plagiarism Detection
**Methods:**
- Cosine Similarity
- Jaccard Similarity
- Containment Measure
- Longest Common Subsequence (LCS)
- **Dataset:** Webis-CPC-11
- **Results:** Highest accuracy with Cosine Similarity

### 6. Fact-Checking
**Models:** BERT, RoBERTa, XLNet
- **Dataset:** FEVER dataset (Fact Extraction and Verification)
- **Evaluation Metrics:** Accuracy, Confusion Matrix





