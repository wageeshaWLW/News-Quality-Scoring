# News-Quality-Scoring

# 📰 AI-Powered News Quality Assessment  

## 📌 Introduction  

With the ever-growing amount of online content, it is essential to assess news article quality without relying on traditional metrics like **interactions and read time**. Many articles use **clickbait titles** or **inflated content** to manipulate engagement, and with the rise of **Generative AI**, the internet is flooded with **low-value, AI-generated content**.  

This project introduces an **NLP-driven framework** to evaluate news articles based on **six key quality criteria**, aiming to provide **objective, human-relevant scores** without relying on pre-existing ranking algorithms.  

---

## 🏗 Methodology  

Each article is analyzed through **six independent NLP-based evaluation tasks**:  

### 1️⃣ **Clickbait Title Detection**  
- Detects **misleading, exaggerated, or vague** titles.  
- Uses **BERT, RoBERTa**, and transformer-based models.  

### 2️⃣ **Information Density Scoring**  
- Measures **meaningful information vs. fluff**.  
- Techniques:  
  - **TF-IDF & Named Entity Recognition (NER)** for factual density.  
  - **Summarization models (T5, BART)** for redundancy analysis.  

### 3️⃣ **Writing Quality Scoring**  
- Evaluates **grammar, readability, and coherence**.  
- Uses:  
  - **Linguistic metrics (Flesch-Kincaid, perplexity analysis)**.  
  - **GPT-4/BLOOM fine-tuned for quality scoring**.  

### 4️⃣ **AI-Generated Content Detection**  
- Distinguishes between **human-written vs. AI-generated** text.  
- Techniques:  
  - **Perplexity-based models** (AI content is often lower in perplexity).  
  - **Fine-tuned BERT/DetectGPT for classification**.  

### 5️⃣ **Plagiarism Detection**  
- Detects **copied or slightly modified** content.  
- Methods:  
  - **TF-IDF & n-gram similarity checks**.  
  - **Semantic similarity using SBERT & Universal Sentence Encoder**.  

### 6️⃣ **Fact-Checking**  
- Verifies statements against **trusted external databases**.  
- Uses:  
  - **Named Entity Recognition (NER) + FEVER-trained models**.  
  - **Retrieval-Augmented Generation (RAG) models for verification**.  

---

## 📁 Project Structure  


