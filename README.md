# Information Retrieval System (BM25 + TF-IDF + Hybrid)

This project implements a fully local IR system built from scratch using:
- BM25 retrieval
- TF–IDF cosine similarity
- A hybrid weighted combination

## Project Structure

IR-System/
│── data/Articles.csv
│── src/
│     ├── preprocess.py
│     ├── bm25_retrieval.py
│     ├── tfidf_retrieval.py
│     ├── combined_retrieval.py
│     ├── evaluation.py
│     └── main.py
├── README.md
└── requirements.txt

## Installation
pip install -r requirements.txt
python -m nltk.downloader stopwords

## Run System
python src/main.py

## Features
✓ BM25 ranking  
✓ TF-IDF ranking  
✓ Hybrid BM25 + TF-IDF  
✓ Evaluation (precision, recall, F1)  
✓ Fully local, reproducible  
