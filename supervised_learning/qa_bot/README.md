# QA-Bot

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project involves creating a question-answering system utilizing BERT for text retrieval, semantic search, and user interaction through a command-line interface. It covers multi-reference question answering, enabling effective querying across a corpus of documents.

## Prerequisites

- Python 3.x
- TensorFlow
- Transformers library
- TensorFlow Hub

## Task Summaries

0. **Question Answering**:
   - File: `0_qa.py`
      Developed a function to find a text snippet in a reference document that answers a user-provided question. This utilizes the `bert-uncased-tf2-qa` model from `tensorflow-hub` and `BertTokenizer` from the `transformers` library.

1. **Create the Loop**:
   - File: `1-loop.py`
      Created a script for user input that prompts with "Q:" and responds with "A:". It allows the user to exit with specific keywords, responding appropriately.

2. **Answer Questions**:
   - File: `2-qa.py`
      Wrote a function that answers questions based on a reference text. If no answer is found, it returns a default message indicating that the question is not understood.

3. **Semantic Search**:
   - File: `3-semantic_search.py`
      Implemented a function that performs semantic search on a corpus of documents to find the most similar text to a given sentence.

4. **Multi-reference Question Answering**:
   - File: `4-qa.py`
      Created a function that answers questions using multiple reference texts, enabling users to query across a larger corpus.
