# 🛍️ Multimodal Price Regressor 💰

An **end-to-end machine learning project** to predict **e-commerce product prices** from **multi-modal data** (text + images).  
This solution employs a **feature fusion strategy**, combining **advanced NLP** features from product descriptions with **computer vision** features from product images.  
An ensemble of **gradient boosting models** is then trained on the combined feature set to perform the final price regression.

---

## 📜 Overview

The goal of this project is to **accurately estimate the price** of an online product given its catalog description and image.  
This is a **classic regression** problem with a **multi-modal twist**.

The solution pipeline includes:
- Data ingestion & cleaning
- Feature engineering
- Memory-efficient processing
- Model training
- Final price prediction

---

## 🛠️ Tech Stack

**Language:** Python 3.11+  

**Core Libraries:**
- 🧮 Data Handling: `Pandas`, `NumPy`
- 🤖 Machine Learning: `Scikit-learn`, `LightGBM`, `XGBoost`
- 🧠 Deep Learning: `TensorFlow`, `PyTorch`, `Transformers`
- 🖼️ Image Processing: `Pillow`
- ⚡ Utilities: `Tqdm`, `Requests`, `Joblib`

**Environment:** Jupyter Notebook, VS Code

---

## 📁 Project Structure
```
multimodal-price-regressor/
│
├── dataset/
│ ├── train.csv
│ └── test.csv
│
├── images/
│ └── (Downloaded product images...)
│
├── notebooks/
│ └── price_prediction.ipynb # Main Jupyter Notebook
│
├── saved_features/
│ ├── distilbert_embeddings.npy
│ ├── X_img_train.npy
│ └── X_img_test.npy
│
├── utils.py # Helper functions (e.g., parallel image downloader)
├── requirements.txt # Project dependencies
└── README.md
---
```
## 🧠 Methodology

The solution is built on a robust pipeline that **processes and combines features** from different data sources before feeding them into a **powerful ensemble model**.

### 1. 🧼 Data Preprocessing
* **Text Cleaning:**  
  - Convert text to lowercase  
  - Remove HTML tags, punctuation, and stopwords  
  - Retain meaningful words for better feature extraction

* **Image Downloading:**  
  - A parallel downloader (`utils.py`) fetches and saves product images  
  - Handles errors, retries, and timeouts efficiently

---

### 2. 🧬 Feature Engineering

#### 📝 Text Features (Hybrid)
* **TF-IDF:** Capture keyword importance and frequency (top 1500 terms)
* **DistilBERT:** Generate dense semantic embeddings for deeper context

#### 🖼️ Image Features
* **ResNet50 (pre-trained):**  
  - Images resized to 224x224  
  - Extract 2048-dimensional visual feature vector

#### 🧭 Feature Reduction (PCA)
* BERT (768 → **128** dimensions)
* ResNet50 (2048 → **256** dimensions)

#### 🔗 Feature Fusion
* Final feature vector = **TF-IDF (1500)** + **BERT (128)** + **Image (256)**

---

### 3. 📊 Modeling & Evaluation

* **Ensemble Models:** `LightGBM` + `XGBoost`
* **Prediction Averaging:** 50/50 average of both models’ outputs
* **Target Transformation:**  
  - `log1p` before training  
  - `expm1` after prediction
* **Evaluation Metric:** SMAPE (Symmetric Mean Absolute Percentage Error)
* **Cross Validation:** 5-Fold CV for stable performance

---

## 🚀 Setup & Usage

### 1. Clone the repository

git clone https://github.com/your-username/multimodal-price-regressor.git
cd multimodal-price-regressor
python -m venv venv
.\venv\Scripts\activate       # (Windows)
# source venv/bin/activate    # (Mac/Linux)

pip install -r requirements.txt
## 🧪 3. Run the Jupyter Notebook

1. **Open the project** in **VS Code**
2. Navigate to:
notebooks/price_prediction.ipynb
3. **Run all cells sequentially** to:
- 📥 Download data
- 🧬 Extract features
- 🤖 Train models
- 📈 Generate predictions

💡 **Note:**  
The first run may take longer due to feature extraction,  
but outputs are saved as `.npy` files for **faster reuse** in future runs.

---

## 🎯 Results

- ✅ **Final Ensemble Model:** Competitive SMAPE Score  
- 🏆 **Best Local CV Score:** `52.xx %` *(example)*

---

## 🔮 Future Scope

- ⚡ **Advanced Ensembling:** Replace simple averaging with **stacking**
- 🧪 **Hyperparameter Tuning:** Use `Optuna` for optimal model parameters
- 🏗️ **Different Architectures:** Try `EfficientNet` for images or other `BERT` variants for text

---

## ✨ Acknowledgments

- 🤝 [Hugging Face Transformers](https://huggingface.co/)
- 🔥 [PyTorch](https://pytorch.org/) & [TensorFlow](https://www.tensorflow.org/) Communities
- 🌿 [LightGBM](https://github.com/microsoft/LightGBM) & [XGBoost](https://github.com/dmlc/xgboost) Contributors

---

## 📝 License

This project is licensed under the **MIT License**.

---

✅ **Tip:** You can also add badges like Python version, build status, or license at the top of your README to make it look even more professional.  

Would you like me to **add badges and a small banner image section** too (like top GitHub repos have)? 🚀

