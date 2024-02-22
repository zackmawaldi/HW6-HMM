# HW6-HMM [![project6](https://github.com/zackmawaldi/HW6-HMM/actions/workflows/test.yml/badge.svg)](https://github.com/zackmawaldi/HW6-HMM/actions/workflows/test.yml)

This package implements the Forward and Viterbi Algorithms (dynamic programming) of Hidden Markov Models (HMMs). 


# Description of methods

The implementation of the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs) are directly followed from the following [resource](https://web.stanford.edu/~jurafsky/slp3/A.pdf). See pages 7 for the forward, and pages 9-10 for Viterbi.

The basic gist for Forward is how can we compute the probability of getting a specific observation sequence.
The basic gist for Viterbi is how can we compute the highest probability-yielding sequence of hidden states that yields a final specific observation sequence.

For the forward, this is essentially done by computing the summed probabilities of sequential states in order.
For the Viterbi, this is essentially done by computing the most probable final outcome, then backtracing sequentially the most probable order.



## Task List

[X] Complete the HiddenMarkovModel Class methods  <br>
  [X] complete the `forward` function in the HiddenMarkovModelClass <br>
  [X] complete the `viterbi` function in the HiddenMarkovModelClass <br>

[X] Unit Testing  <br>
  [X] Ensure functionality on mini and full weather dataset <br>
  [X] Account for edge cases 

[X] Packaging <br>
  [X] Update README with description of your methods <br>
  [X] pip installable module (optional)<br>
  [X] github actions (install + pytest) (optional)