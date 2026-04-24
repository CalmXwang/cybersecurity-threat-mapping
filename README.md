# Cybersecurity Threat Mapping System

## Overview
This project builds a pipeline to map CVEs to MITRE ATT&CK techniques using NLP and ranking methods.

The goal is to support cybersecurity analysis by connecting vulnerability information with real-world attack behaviors.

## Features
- CVE data processing
- Text cleaning and preprocessing
- ATT&CK technique mapping
- Ranking using BM25
- Result evaluation

## Tech Stack
Python, NLP, Pandas, BM25

## Why This Project Matters
This project helps automate the process of analyzing vulnerabilities and understanding possible attack patterns, which is important in real-world cybersecurity.

## What I Learned
- Working with real-world data
- Handling unstructured text
- Building data pipelines
- Debugging and improving system performance

## How to Run

1. Install dependencies:

``bash
pip install -r requirements.txt
Run the main script:
python main.py
Optional: run tests:
python run_tests.py

## Project Structure

``text
cybersecurity-threat-mapping/
├── README.md
├── requirements.txt
├── main.py
├── run_tests.py
├── live_data.py
├── sysml_to_cpe.py
├── sysml_cpe.py
├── Phase6-1_data_prep.py
├── Phase6-2_biencoder.py
├── Phase6-3_classifier.py
├── Phase6-4_soi_reranker.py
├── Phase6-5_eval.py
└── Phase6-6_run_pipeline.py
