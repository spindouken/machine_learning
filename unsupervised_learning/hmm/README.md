### Hidden Markov Models (Implemented from Scratch Using NumPy)

#### Project Summary

This project focuses on Hidden Markov Models (HMM) and covers various concepts related to Markov chains and algorithms used for HMMs, including:

- Markov Chain Probability Calculation
- Steady State Probabilities for Regular Markov Chains
- Identification of Absorbing Chains
- The Forward Algorithm for HMMs
- The Viterbi Algorithm for State Sequence Prediction
- The Backward Algorithm for HMMs

---

#### Task Summaries

0. **Markov Chain**
    - Implements a function to calculate the probability of a Markov chain being in a specific state after a given number of iterations using a transition matrix and initial state probabilities. Utilizes NumPy for matrix operations.

1. **Regular Chains**
    - Implements a function to determine the steady state probabilities of a regular Markov chain based on the transition matrix. Uses NumPy for calculations.

2. **Absorbing Chains**
    - Implements a function to determine if a given Markov chain is absorbing by analyzing its transition matrix. Utilizes NumPy.

3. **The Forward Algorithm**
    - Implements the forward algorithm for a hidden Markov model, calculating the likelihood of observations and the forward path probabilities using emission and transition probabilities. Utilizes NumPy.

4. **The Viterbi Algorithm**
    - Implements the Viterbi algorithm to find the most likely sequence of hidden states given a series of observations, emission probabilities, and transition probabilities. Utilizes NumPy.

5. **The Backward Algorithm**
    - Implements the backward algorithm for a hidden Markov model, calculating the likelihood of observations and backward path probabilities using the same foundational parameters as the forward algorithm. Utilizes NumPy.
