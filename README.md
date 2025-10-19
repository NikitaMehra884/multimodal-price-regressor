# multimodal-price-regressor
Multimodal Price Regressor 🛍️💰
An end-to-end machine learning project to predict e-commerce product prices from multi-modal data (text and images). This solution employs a feature fusion strategy, combining advanced NLP features from product descriptions with computer vision features from product images. An ensemble of gradient boosting models is then trained on the combined feature set to perform the final price regression.

Overview 📜
The goal of this project is to accurately estimate the price of an online product given its catalog description and image. This is a classic regression problem with a multi-modal twist. The solution pipeline handles everything from data ingestion and cleaning to advanced feature engineering, memory-efficient processing, model training, and final prediction.

Tech Stack 🛠️
Language: Python 3.11+

Core Libraries:

Data Handling: Pandas, NumPy

Machine Learning: Scikit-learn, LightGBM, XGBoost

Deep Learning (for features): TensorFlow, PyTorch, Transformers

Image Processing: Pillow

Utilities: Tqdm, Requests, Joblib

Environment: Jupyter Notebook, VS Code

Project Structure 📁
multimodal-price-regressor/
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── images/
│   ├── (Downloaded product images...)
│
├── notebooks/
│   └── price_prediction.ipynb   # Main Jupyter Notebook
│
├── saved_features/
│   ├── distilbert_embeddings.npy
│   ├── X_img_train.npy
│   └── X_img_test.npy
│
├── utils.py                     # Helper functions (e.g., parallel image downloader)
├── requirements.txt             # Project dependencies
└── README.md                    # You are here
Methodology 🧠
The solution is built on a robust pipeline that processes and combines features from different data sources before feeding them into a powerful ensemble model.

1. Data Preprocessing
Text Cleaning: The raw catalog_content is cleaned by converting it to lowercase, removing HTML tags, punctuation, and common English stopwords. This step ensures that only meaningful words are used for feature extraction.

Image Downloading: A parallel image downloader script (utils.py) efficiently fetches and saves all product images from their URLs into a local images/ directory. The downloader is robust, handling potential network errors and timeouts with automatic retries.

2. Feature Engineering
This is the core of the project, where we extract powerful numerical representations from raw data.

Text Features (Hybrid Approach):

TF-IDF Features: We use TfidfVectorizer to capture keyword importance and frequency, creating a sparse matrix of the top 1500 terms. This provides a strong baseline signal.

Semantic Features (DistilBERT): We use a pre-trained DistilBERT model to generate dense embeddings (vectors) for each product description. These embeddings capture the semantic meaning and context of the text, providing a much deeper understanding than TF-IDF alone.

Image Features (Deep Learning):

A pre-trained ResNet50 model is used as a feature extractor.

Each image is resized to 224x224, preprocessed, and fed into the model to produce a 2048-dimensional vector representing its visual characteristics.

Feature Reduction (Memory Optimization):

To manage the high memory requirements of large feature sets (especially on an 8GB RAM system), Principal Component Analysis (PCA) is applied.

DistilBERT embeddings (768 dimensions) are reduced to the most important 128 dimensions.

ResNet50 image features (2048 dimensions) are reduced to the most important 256 dimensions.

Feature Fusion:

The final feature set for the model is created by combining the three types of features: TF-IDF (1500) + Reduced BERT (128) + Reduced Image (256).

This hybrid feature vector provides the model with a rich, comprehensive understanding of each product.

3. Modeling and Evaluation
Ensemble Modeling: To maximize accuracy, we use an ensemble of two powerful gradient boosting models: LightGBM and XGBoost. Both models are trained on the final fused feature set.

Prediction Averaging: The final price prediction is a simple 50/50 average of the predictions from the LightGBM and XGBoost models. This technique often improves robustness and reduces individual model biases.

Target Transformation: The price variable has a skewed distribution. We apply a log1p transformation before training and an expm1 transformation after prediction to improve model performance.

Evaluation: The model's performance is validated using the SMAPE (Symmetric Mean Absolute Percentage Error) metric, tracked via a robust 5-Fold Cross-Validation strategy to get a reliable estimate of the model's true performance.

Setup and Usage 🚀
Clone the repository:

Bash

git clone https://github.com/your-username/multimodal-price-regressor.git
cd multimodal-price-regressor
Create a virtual environment and install dependencies:

Bash

python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
Run the Jupyter Notebook:

Open the project in VS Code.

Navigate to the notebooks/price_prediction.ipynb file.

Run the cells sequentially to execute the entire pipeline, from data download to final submission file generation. Note: The initial feature extraction steps will take a long time but will save their results to .npy files for instant loading in future sessions.

Results 🎯
The final ensemble model achieved a competitive SMAPE score by effectively leveraging the strengths of different feature types and models.

Best Local CV Score: [Your best average CV SMAPE score, e.g., 52.xx %]

Future Scope 🔮
Advanced Ensembling: Implement stacking instead of simple averaging for the final ensemble.

Hyperparameter Tuning: Use a more advanced tool like Optuna to find the optimal settings for both LightGBM and XGBoost.

Different Architectures: Experiment with different pre-trained models for feature extraction, such as EfficientNet for images or other BERT variants for text.
