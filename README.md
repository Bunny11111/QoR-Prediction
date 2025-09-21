# ML for EDA – Area Prediction & Recipe Optimization

This project explores **Machine Learning for Electronic Design Automation (EDA)**, focusing on two main goals:

1. **Predicting Quality-of-Results (QoR) — specifically, circuit area — before running full synthesis**
2. **Generating efficient synthesis recipes automatically using Reinforcement Learning (RL)**

By leveraging **Graph Neural Networks (GNNs)**, **LSTMs**, and **RL**, this project demonstrates how AI can accelerate the chip design process, reduce iterations, and improve design efficiency.

---

## 📌 Problem Statement

Given:

* A **digital circuit netlist** (in `.bench` format)
* A **synthesis recipe** (sequence of ABC commands)

Predict:

* The **area** after applying the recipe (QoR metric)

Bonus Task:

* Use RL to **discover optimal synthesis recipes** that minimize area automatically.

---

## 📂 Dataset

* **Design Benchmarks:**

  * `aes_orig.bench` (large)
  * `tv80_orig.bench` (medium)
  * `i2c_orig.bench` (small)
* **Recipe Generation:** Random sequences of 12 possible synthesis commands
* **Data Points:** \~3000 total (1000 per design)
* **Labels:** Synthesized area (normalized to \[0, 1000])

---

## 🧠 Approach

### 🔹 Preprocessing

* **Circuit Representation:**

  * Parse `.bench` files and construct a graph (nodes = gates/inputs, edges = connectivity)
  * Encode nodes as one-hot vectors → PyTorch Geometric `Data` object

* **Recipe Encoding:**

  * Tokenize recipe string (split by `;`)
  * Convert to token IDs and embed for input to LSTM

* **Area Normalization:**

  * Normalize areas across designs for stable training

### 🔹 Model Architecture

**Hybrid GNN + LSTM Model:**

* **Graph Branch:**

  * GCNConv → ReLU → GCNConv → ReLU → Global Mean Pooling
* **Recipe Branch:**

  * Embedding Layer → LSTM → Final hidden state used as recipe embedding
* **Fusion + Regression:**

  * Concatenate embeddings → MLP (128 → 64 → 1) → Predicted Area

**Loss Function:** MSE
**Optimizer:** Adam

### 🔹 Training & Fine-Tuning

* Phase 1: Train on Design 1
* Phase 2: Fine-tune on Design 1 + 2
* Phase 3: Fine-tune on all three designs for better generalization

### 🔹 Reinforcement Learning for Recipe Optimization

* **Environment:** ABC synthesis tool
* **Agent:** Policy network selecting next synthesis command
* **Reward:** Area reduction after command execution
* **Algorithm:** REINFORCE-style policy gradient with Adam optimizer

---

## 📊 Results

| Design   | RMSE After Fine-Tuning |
| -------- | ---------------------- |
| Design 1 | \~47 (2–3% diff)       |
| Design 2 | \~525 (2–3% diff)      |
| Design 3 | \~3086 (5–6% diff)     |

* Fine-tuning significantly reduced error for unseen designs
* RL agent successfully discovered efficient recipes leading to area reduction

---

## 🚀 Future Work

* Improve dataset balance and normalization
* Explore design-specific embeddings for better generalization
* Extend RL approach to multi-objective optimization (e.g., area + timing + power)

---

## 🛠️ Tech Stack

* **Python**
* **PyTorch + PyTorch Geometric** (for GNNs)
* **ABC Logic Synthesis Tool**
* **Reinforcement Learning (Policy Gradient)**

---

## 👨‍💻 Contributors

* **Kotha Hitesh** (220101058)
* **Bonda Jnana Sai** (220101025)
* **Burada Jitendra Sai** (220101028)
* **G.K.S. Deepak** (220101043)
