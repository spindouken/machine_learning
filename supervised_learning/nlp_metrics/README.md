# NLP - Evaluation Metrics (N-GRAM BLEU, ROUGE)

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Task Summaries](#task-summaries)

## Project Overview

This project focuses on calculating BLEU (Bilingual Evaluation Understudy) scores for evaluating the quality of machine-generated translations against reference translations. It covers the computation of unigram BLEU, n-gram BLEU, and cumulative n-gram BLEU scores, which are essential metrics in natural language processing for assessing translation accuracy.

## Prerequisites

- Python
- NumPy (for numerical operations)

## Task Summaries

0. **Unigram BLEU Score**: 
   - File: `0-uni_bleu.py`
   - This script implements the function `uni_bleu`, which calculates the unigram BLEU score for a given sentence based on a list of reference translations. It evaluates how many unigrams (individual words) from the generated sentence match those in the references. The implementation is done from scratch, using basic string and list operations to compute the score.

1. **N-gram BLEU Score**: 
   - File: `1-ngram_bleu.py`
   - This script contains the function `ngram_bleu`, which computes the n-gram BLEU score for a sentence using a specified n-gram size. The function compares n-grams of the generated sentence with those in the reference translations, returning the score. This task utilizes custom logic to extract n-grams from sentences, ensuring that the implementation is entirely from scratch.

2. **Cumulative N-gram BLEU Score**: 
   - File: `2-cumulative_bleu.py`
   - This script implements the function `cumulative_bleu`, which calculates the cumulative n-gram BLEU score for a sentence using the largest specified n-gram size. This function averages the BLEU scores across all n-grams from 1 to n, providing a comprehensive evaluation of translation quality. The averaging and score calculations are performed using custom logic without relying on external libraries.
