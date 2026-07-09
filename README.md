# Self-Attention from Scratch using NumPy

## Overview

This project is my first step in building a strong foundation for **Large Language Models (LLMs)** and **Transformer-based architectures**.
In this implementation, I built the **Self-Attention mechanism from scratch using Python and NumPy**, without using deep learning libraries such as TensorFlow or PyTorch.

Self-Attention is one of the most important ideas behind modern LLMs. It allows each token in a sequence to look at other tokens in the same sequence and decide **which tokens are more important for understanding context**.

The goal of this project is not just to get an output, but to understand the **full mathematical and implementation flow** of self-attention step by step.

---

## Why this project?

Large Language Models like GPT, BERT, and many Transformer-based systems rely heavily on **attention mechanisms**. Before working on advanced LLM projects, I wanted to understand the most fundamental building block of Transformers at a low level.

This project helped me learn:

* how tokens are represented as vectors
* how **Query, Key, and Value** matrices are created
* how similarity between tokens is measured
* how attention weights are computed
* how the final context-aware representation is generated

This is part of my semester-long LLM learning journey, where I am building concepts from scratch and gradually moving toward a **portfolio-level LLM project**.

---

## Learning Objective

The objective of this project is to understand and implement the **Self-Attention mechanism manually** using NumPy.

By completing this task, I aimed to:

* understand the intuition behind self-attention
* implement the mathematical operations step by step
* learn how **Q, K, and V** are generated from input embeddings
* compute attention scores using matrix multiplication
* apply scaling and softmax correctly
* generate the final attention output
* build a strong base for future topics like:

  * Multi-Head Attention
  * Positional Encoding
  * Transformer Encoder
  * Mini LLM pipeline implementation

---

## What is Self-Attention?

Self-Attention is a mechanism that helps a model understand **relationships between words/tokens in the same input sequence**.

For example, in a sentence like:

> **"The cat sat on the mat because it was soft."**

the word **"it"** should pay attention to **"mat"** more than unrelated words.
Self-attention helps the model decide this mathematically.

Instead of processing tokens one by one like traditional sequence models, self-attention allows **every token to interact with every other token** and compute a context-aware representation.

---

## Core Idea

For every token embedding in the input sequence, we create three vectors:

* **Query (Q)** → what this token is looking for
* **Key (K)** → what this token contains / offers
* **Value (V)** → the actual information carried by the token

Then the model does the following:

1. Compare each token’s **Query** with all tokens’ **Keys**
2. Compute similarity scores
3. Convert those scores into probabilities using **Softmax**
4. Use those probabilities to combine the **Values**
5. Produce a new representation for each token that contains context from the whole sequence

---

## Mathematical Workflow

If the input embedding matrix is:

[
X \in \mathbb{R}^{n \times d}
]

where:

* **n** = number of tokens
* **d** = embedding dimension

then we create three trainable weight matrices:

[
W_Q,; W_K,; W_V
]

and compute:

[
Q = XW_Q
]

[
K = XW_K
]

[
V = XW_V
]

### Step 1: Attention Scores

We compare each query with all keys:

[
\text{Scores} = QK^T
]

This gives a score matrix showing how much each token should attend to every other token.

---

### Step 2: Scaling

To avoid very large values when the dimension grows, we scale the scores:

[
\text{Scaled Scores} = \frac{QK^T}{\sqrt{d_k}}
]

where:

* (d_k) = dimension of the key vectors

---

### Step 3: Softmax

We convert the scaled scores into probabilities:

[
\text{Attention Weights} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
]

Each row of the attention weights matrix sums to **1**.

---

### Step 4: Final Attention Output

The attention weights are used to combine the value vectors:

[
\text{Output} = \text{Attention Weights} \cdot V
]

This output is the **context-aware representation** of the input sequence.

---

## Project Workflow

This implementation follows the self-attention mechanism step by step.

### 1. Define Input Embeddings

Create a small matrix representing token embeddings for a sample sequence.

### 2. Initialize Weight Matrices

Initialize three matrices:

* Query weight matrix (W_Q)
* Key weight matrix (W_K)
* Value weight matrix (W_V)

These matrices are used to project the input embeddings into Q, K, and V spaces.

### 3. Generate Query, Key, and Value Matrices

Using matrix multiplication:

* (Q = XW_Q)
* (K = XW_K)
* (V = XW_V)

### 4. Compute Raw Attention Scores

Multiply Query with the transpose of Key:

* (QK^T)

This produces token-to-token similarity scores.

### 5. Scale the Scores

Divide the scores by (\sqrt{d_k}) to stabilize the values.

### 6. Apply Softmax

Convert the scaled scores into attention probabilities.

### 7. Compute Final Attention Output

Multiply the attention weights by the Value matrix to obtain the final output.

---

## Example Flow

Suppose we have 3 tokens in a sentence.
Each token is represented by an embedding vector.

The self-attention mechanism allows:

* token 1 to look at token 1, token 2, token 3
* token 2 to look at token 1, token 2, token 3
* token 3 to look at token 1, token 2, token 3

This means every token gets a chance to collect useful contextual information from the full sequence.

That is what makes Transformers so powerful for language tasks.

---

## Technologies Used

* **Python 3**
* **NumPy**
* **Jupyter Notebook / Python Script**

---

## Project Structure

```bash
self-attention-numpy/
│── self_attention.ipynb
│── self_attention.py        # optional script version
│── README.md
```

If you only have the notebook for now, you can keep it like this:

```bash
self-attention-numpy/
│── self_attention.ipynb
│── README.md
```

---

## Output Generated

The implementation prints / shows the following intermediate and final outputs:

* Input embedding matrix
* Query matrix (**Q**)
* Key matrix (**K**)
* Value matrix (**V**)
* Raw attention scores
* Scaled attention scores
* Attention weight matrix
* Final self-attention output

These outputs make it easier to understand how data moves through each stage of the attention mechanism.

---

## Key Concepts Covered

This project covers the following LLM and Transformer fundamentals:

* Self-Attention
* Scaled Dot-Product Attention
* Query, Key, and Value (QKV)
* Token interaction inside a sequence
* Matrix multiplication in attention computation
* Softmax normalization
* Context-aware token representation
* Foundation of Transformer models

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/self-attention-numpy.git
cd self-attention-numpy
```

### 2. Install NumPy

```bash
pip install numpy
```

### 3. Run the implementation

If using a Python script:

```bash
python self_attention.py
```

If using Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```bash
self_attention.ipynb
```

---

## What I Learned from this Project

Through this project, I learned how self-attention works internally rather than treating it as a black box.

### Technical understanding gained:

* how embeddings are transformed into Q, K, and V
* how attention scores are calculated
* why scaling by (\sqrt{d_k}) is necessary
* how softmax turns scores into meaningful weights
* how the weighted sum of values creates context-aware output

### Conceptual understanding gained:

* why attention is better at capturing long-range dependencies
* how Transformers process all tokens in parallel
* why self-attention is the foundation of modern LLMs

---

## Limitations of the Current Version

This project is intentionally a **basic single-head self-attention implementation** for learning purposes.

Current limitations:

* no positional encoding
* no masking
* no multi-head attention
* no trainable optimization loop
* no PyTorch / TensorFlow implementation
* uses small manual embeddings for demonstration

---

## Future Improvements

This project is the starting point of a larger LLM-focused learning roadmap.
Planned future extensions include:

* **Multi-Head Attention from scratch**
* **Positional Encoding implementation**
* **Layer Normalization**
* **Residual Connections**
* **Transformer Encoder block**
* **Mini text generation / LLM-style toy project**
* **PyTorch version of self-attention**
* **Visualization of attention weights**
* **End-to-end portfolio LLM project on GitHub**

---

## Semester LLM Project Roadmap

This repository is part of a broader semester project where I will gradually move from **fundamentals to a portfolio-level LLM project**.

Planned progression:

1. Self-Attention using NumPy
2. Multi-Head Attention
3. Positional Encoding
4. Transformer Encoder block
5. Tokenization and embeddings pipeline
6. Small language modeling experiments
7. End-to-end LLM-inspired project for GitHub portfolio

---

## Repository Purpose

This repository is meant to document my practical understanding of LLM fundamentals by implementing core ideas from scratch.

Instead of only using high-level libraries, I want to understand:

* the mathematics behind Transformer components
* how these components are implemented in code
* how to explain them clearly
* how to build toward real LLM applications

---

## Author

**Rupa Sivakumar**
Second-Year Computer Science Student
Interested in **LLMs, Python, AI projects, and building portfolio-level systems from fundamentals**

---

## Final Note

This project is a foundational step in my LLM journey.
Understanding self-attention deeply is essential before moving into larger Transformer and LLM implementations.
This repository represents not just a coding task, but a deliberate effort to learn **how modern language models actually work under the hood**.
