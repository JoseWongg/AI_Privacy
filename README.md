# Sentiment Analysis with Federated Learning & Differential Privacy

Train a privacy-preserving **LSTM** sentiment classifier on customer reviews. The model is trained with **TensorFlow Federated (TFF)** and a **Differential Privacy** optimiser (gradient clipping + noise), so data stays on clients and individual privacy is protected.

---

## What this project does
- Classifies reviews as **positive/negative** using their text.
- Simulates **federated learning with 5 clients** via TFF.
- Applies **DP-SGD** (clipping + Gaussian noise) on client updates.
- Reports performance metrics and a privacy budget (ε at δ=1e-5).

---

## Tech stack
Python • TensorFlow • TensorFlow Federated • TensorFlow Privacy • scikit-learn • Pandas • NumPy • Matplotlib • NLTK • Gensim

---

## Dataset
- **Rows:** 21,966 reviews  
- **Columns:** `name`, `country`, `date_time`, `stars`, `review_head`, `review_body`
- **Used by the model:**  
  - `review_body` → input text  
  - `stars` → converted to binary labels (e.g., ≥3 = positive)


---

## Results
- Model: `Embedding → LSTM → LSTM → Dense(sigmoid)`
  - Accuracy: **75.69%**
  - F1-score: **0.86**
  - Privacy budget: **ε ≈ 75.67** at **δ = 1e-5**


## How to Run  

1. Clone this repository:  
   ```bash
   git clone https://github.com/JoseWongg/AI_Privacy.git
   cd AI_Privacy

