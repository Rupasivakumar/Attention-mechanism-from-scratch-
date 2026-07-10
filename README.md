# Attention Mechanism using NumPy

## 📌 Project Overview

This project demonstrates the implementation of the **Attention Mechanism** from scratch using **Python** and **NumPy**, without relying on deep learning libraries such as TensorFlow or PyTorch.

Attention is one of the most important ideas in modern deep learning and is a foundational concept behind **Large Language Models (LLMs)** and **Transformer-based architectures**. It helps a model decide **which parts of the input are more important** while generating an output or understanding context.

This project was built as part of my semester-long LLM learning journey to understand the mathematical and implementation-level details of attention before moving toward advanced topics such as **self-attention, multi-head attention, transformers, and LLMs**.

---

## 🎯 Objective

The main goals of this project are:

* To understand the **core idea of the Attention Mechanism**
* To implement attention computations manually using **NumPy**
* To learn how **Query (Q), Key (K), and Value (V)** representations are used
* To understand how attention scores are calculated and normalized
* To build a strong foundation for future **LLM and Transformer projects**

---

## 🧠 Why Attention Matters

Traditional sequence models often struggle to capture long-range relationships between words or tokens in a sequence. The Attention Mechanism solves this by allowing the model to **focus on the most relevant parts of the input** while processing information.

In simple terms, attention answers the question:

> **“Out of all the available input tokens, which ones should the model focus on more right now?”**

This concept is at the heart of:

* Machine Translation
* Text Summarization
* Question Answering
* Transformers
* Large Language Models (LLMs)

---

## 🛠️ Technologies Used

* **Python 3**
* **NumPy**
* **Jupyter Notebook**

---

## 📂 Project Structure

```bash
Attention-Mechanism/
│── attention_mechanism.ipynb
│── README.md
```

> If converted into a Python script later:

```bash
Attention-Mechanism/
│── attention_mechanism.py
│── README.md
```

---

## ⚙️ Concepts Implemented

This project covers the basic workflow of the **Attention Mechanism**:

1. Creating input embeddings
2. Initializing weight matrices
3. Generating **Query (Q)**, **Key (K)**, and **Value (V)** matrices
4. Computing attention scores
5. Scaling the scores
6. Applying **Softmax**
7. Generating the final attention output

---

## 🔍 Step-by-Step Working

### Step 1: Input Embeddings

We begin with a set of input vectors (embeddings), where each row represents a token and each column represents a feature.

Example:

```python
X = [
  [x11, x12, x13, ...],
  [x21, x22, x23, ...],
  [x31, x32, x33, ...]
]
```

These embeddings are the numerical representations of input tokens.

---

### Step 2: Initialize Weight Matrices

Three trainable weight matrices are used:

* **WQ** → Query weight matrix
* **WK** → Key weight matrix
* **WV** → Value weight matrix

These matrices transform the input embeddings into three different representations required for attention.

---

### Step 3: Compute Query, Key, and Value Matrices

The input embeddings are multiplied with the weight matrices:

```python
Q = X @ WQ
K = X @ WK
V = X @ WV
```

Where:

* **Q (Query)** represents what a token is looking for
* **K (Key)** represents what a token contains
* **V (Value)** represents the actual information carried by the token

---

### Step 4: Calculate Attention Scores

The attention score is calculated using the dot product of **Q** and **Kᵀ**:

```python
scores = Q @ K.T
```

This gives a score showing how strongly one token should attend to another.

---

### Step 5: Scale the Scores

The raw attention scores are scaled by the square root of the key dimension:

```python
scaled_scores = scores / sqrt(dk)
```

where **dk** is the dimension of the key vectors.

Scaling helps stabilize the values before applying Softmax and prevents extremely large scores.

---

### Step 6: Apply Softmax

Softmax converts the scaled scores into probabilities:

```python
attention_weights = softmax(scaled_scores)
```

These probabilities represent how much focus each token should place on the others.

---

### Step 7: Compute Final Attention Output

Finally, the attention weights are multiplied with the Value matrix:

```python
output = attention_weights @ V
```

This produces the final **attention output**, which is a context-aware representation of the input.

---

## 🧮 Mathematical Formulation

The Attention Mechanism can be summarized as:

```math
Q = XW_Q
K = XW_K
V = XW_V
```

```math
\text{Scores} = QK^T
```

```math
\text{Scaled Scores} = \frac{QK^T}{\sqrt{d_k}}
```

```math
\text{Attention Weights} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
```

```math
\text{Output} = \text{Attention Weights} \cdot V
```

---

## 📊 Output of the Program

The notebook/program displays the following intermediate and final results:

* Input Embeddings
* Query Matrix (**Q**)
* Key Matrix (**K**)
* Value Matrix (**V**)
* Raw Attention Scores
* Scaled Attention Scores
* Attention Weights
* Final Attention Output

This makes it easier to understand how each mathematical step contributes to the final result.

---

## 📘 Key Concepts Learned

Through this project, I explored:

* Attention Mechanism
* Query, Key, Value (QKV)
* Matrix Multiplication in NumPy
* Scaled Dot-Product Attention
* Softmax Normalization
* Context-based token weighting
* Foundations of Transformer architecture
* Core concepts behind LLMs

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/attention-mechanism-numpy.git
```

### 2. Move into the project folder

```bash
cd attention-mechanism-numpy
```

### 3. Install dependencies

```bash
pip install numpy
```

### 4. Open the notebook

If you are using Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```bash
attention_mechanism.ipynb
```

---

## 🎓 Learning Outcome

After completing this project, I gained a practical understanding of:

* How attention works mathematically
* Why Q, K, and V are needed
* How models decide which tokens are more important
* How attention weights are calculated
* How attention creates context-aware representations
* Why attention is the foundation of Transformers and LLMs

---

## 🚀 Future Improvements

This project is the first step in a larger LLM-focused learning roadmap. Possible next improvements include:

* **Self-Attention implementation**
* **Multi-Head Attention**
* **Positional Encoding**
* **Masked Attention**
* **Transformer Encoder block**
* **Mini Transformer from scratch**
* **Visualization of attention weights**
* **PyTorch implementation for comparison**
* **LLM mini-project based on Transformer architecture**

---

## 👩‍💻 Author

**Your Name**

Second-year Computer Science student exploring **LLMs, Deep Learning, DBMS, Python, and AI system development** through hands-on projects.

---

## ⭐ Note

This project is part of my semester-long journey to build a **portfolio-level LLM project** from scratch by first understanding the fundamental building blocks such as **Attention, Transformers, and token representations** before moving to larger implementations.
