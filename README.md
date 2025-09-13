# 🎯 Multimodal Deep Learning for Predicting Parkinson’s Severity (UPDRS)

> **Short**: A multimodal model that fuses voice-derived tabular features and signal-derived images to predict the UPDRS score (Parkinson’s disease severity).

---

## ✨ Overview

This repository contains a PyTorch-based multimodal pipeline that:

* Uses **tabular voice features** (jitter, shimmer, NHR, HNR, etc.) and **image-like representations** of signals.
* Trains a **fusion neural network** (tabular branch + CNN branch → concatenation → FC layers) to **predict UPDRS** (a clinical severity score).

The goal is *severity estimation* (regression), not disease diagnosis.

---

## 📌 Key Concepts

* **UPDRS** — Unified Parkinson’s Disease Rating Scale. Higher values = more severe symptoms.
* **MAE** — Mean Absolute Error (average absolute difference between predicted and true UPDRS).
* **RMSE** — Root Mean Squared Error (penalizes larger errors more than MAE).

Final notebook metrics (example): **MAE ≈ 7.9**, **RMSE ≈ 9.8** (your run may differ).

---

## 🧩 Use Cases

* **Clinical decision support**: auxiliary estimates of symptom severity.
* **Remote monitoring**: estimate progression from voice recordings at home.
* **Research/Trials**: objective, repeatable severity measurements for cohorts.

---

## ⚙️ Project Structure (example)

```
├─ data/
│  ├─ raw/                  # raw csvs / audio files
│  ├─ processed/            # generated images, cleaned csvs
├─ notebooks/
│  └─ parkinson.ipynb       # main notebook (your uploaded file)
├─ src/
│  ├─ dataset.py            # Dataset & transforms
│  ├─ model.py              # MultiModalNet implementation
│  ├─ train.py              # training loop + hyperparam search
│  └─ utils.py              # helpers, metrics
├─ requirements.txt
└─ README.md
```

---

## 🛠️ Installation

1. Create & activate a virtualenv (recommended):

```bash
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate    # Windows
```

2. Install requirements:

```bash
pip install -r requirements.txt
# or
pip install pandas numpy matplotlib torch torchvision scikit-learn
```

---

## 🚀 Quick Start — Run Notebook

1. Place your datasets under `data/raw/` (CSV + optional audio).
2. Open the notebook `notebooks/parkinson.ipynb` and run cells sequentially.
3. The notebook will:

   * preprocess features
   * generate image-like representations (saved to `data/processed/`)
   * train the multimodal model
   * report MAE / RMSE on a validation split

---

## 🧪 Training (script)

Example command for training with best hyperparameters found in the notebook:

```bash
python src/train.py --lr 1e-4 --batch_size 32 --epochs 15 --device cpu
```

### Important training tips

* ✅ **Normalize/scale tabular features** (StandardScaler / MinMax) before training.
* ✅ **Check for NaNs / Infs**: NaN losses usually come from unscaled input, exploding gradients, or invalid targets.
* ⚠️ If you see `Loss: nan`: try lowering the learning rate (e.g. 1e-4 → 1e-5), add gradient clipping, or check inputs for NaN.

---

## 📈 Evaluation

Metrics reported in the notebook:

* **MAE** — lower is better; average absolute error in UPDRS points.
* **RMSE** — lower is better; more sensitive to large errors.

Example output: `Validation MAE: 7.9093`, `Validation RMSE: 9.8411`.

---

## 🔧 How to extend / next steps

* **Classification add-on**: to *detect* Parkinson’s vs Healthy, train a classifier (binary cross-entropy) using labels.
* **Data augmentation**: augment audio or image-like features to improve generalization.
* **Better architectures**: try ResNet-style CNN, attention, or tabular transformers.
* **Cross-validation**: use k-fold CV to get more robust error estimates.
* **Model explainability**: use SHAP / LIME to analyze which features influence UPDRS.

---

## 📝 Notes & Caveats

* This model estimates severity — it is **not a clinical diagnostic tool**. Always consult clinicians for medical decisions.
* UPDRS labeling quality and dataset bias strongly affect model performance.

---

## 📫 Contact

If you want help improving the project, debugging NaNs, or adding a classifier: reach out at `your-email@example.com` or open an issue.

---

## 📜 License

Choose a license (e.g., MIT) and add `LICENSE` file as needed.

---

*Made with ❤️ and science.*
