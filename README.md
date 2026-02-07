# Parallel Random Forest: Design, Implementation, and Evaluation

## 📌 Overview

This project implements a **Random Forest classifier from scratch** with multiple **parallelization strategies** and evaluates their performance in terms of **runtime speedup** and **classification quality**.

The goal is to study **how different parallelization approaches behave in practice**, especially under:
- Large datasets
- Class imbalance
- Realistic preprocessing overheads

The implementation avoids using `sklearn`’s built-in `RandomForestClassifier` for training and instead builds trees manually using `DecisionTreeClassifier`.

---

## 🚀 Parallelization Strategies Implemented

1. **Sequential**
2. **Tree Parallelism**
3. **Data Parallelism**
4. **Hybrid Parallelism**

Each approach builds the **same number of trees**, enabling fair comparison.

---

## 📂 Project Structure

```
.
├── group_63_parallel_random_forest.py
├── rf_training.log
├── outputs/
    ├── confusion_matrix.png
    ├── rf_performance_comparison.png
├── README.md
```

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Macro and weighted averages

Support values are shown **only for individual classes**.  
Accuracy is reported separately.

---

## 📝 Logging

- Uses Python `logging`
- Single shared log file
- Timestamped entries

---

## ▶️ How to Run

```bash
python3 group_63_parallel_random_forest.py
```

Dependencies:
- numpy
- pandas
- scikit-learn

