
# 🚀 **Machine Learning and Deep Learning Projects Repository**

## 📋 **Overview**
This repository is a collection of **Machine Learning (ML)** and **Deep Learning (DL)** projects focusing on data analysis, model building, and evaluation. The projects cover clustering, classification, regression, text generation, and sentiment analysis using various datasets and algorithms.

---

## 📁 **Projects in the Repository**

1. [Census Solution: Clustering on Census Income Dataset](#census-solution)
2. [Deep Learning using Fashion MNIST](#fashion-mnist-solution)
3. [Lending Club Loan Data Analysis](#lending-club-loan-analysis)
4. [Lyrics Generation using LSTM](#lyrics-generation)
5. [Mercedes-Benz Greener Manufacturing](#mercedes-benz-greener-manufacturing)
6. [Movielens Case Study](#movielens-case-study)
7. [Sentiment Detection on IMDB Dataset](#sentiment-detection-imdb)

---

## 🔎 **Project Details**

---

### 1️⃣ **Census Solution: Clustering on Census Income Dataset**
**Objective**: Analyze the Census Income dataset and perform clustering using **KMeans** to uncover patterns in demographic attributes.

**Steps**:
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Scaling and Encoding
- Clustering with KMeans
- Insights and Interpretation

**Tools & Libraries**: Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn

---

### 2️⃣ **Deep Learning using Fashion MNIST**
**Objective**: Build a **Deep Neural Network (DNN)** to classify clothing images from the **Fashion MNIST Dataset** into 10 categories.

**Steps**:
- Data Preprocessing: Normalization and Label Encoding
- Model Development:
  - Input Layer: Flattened 28x28 images
  - Hidden Layers: ReLU Activation
  - Output Layer: Softmax Activation
- Performance Metrics: Accuracy, Confusion Matrix

**Tools & Libraries**: TensorFlow, Keras, Pandas, NumPy, Matplotlib

---

### 3️⃣ **Lending Club Loan Data Analysis**
**Objective**: Analyze Lending Club loan data and predict loan repayment status using a **Neural Network**.

**Steps**:
- Data Cleaning and Feature Engineering
- Exploratory Data Analysis (EDA)
- Neural Network Implementation
- Model Training and Evaluation

**Tools & Libraries**: TensorFlow, Keras, Pandas, Seaborn, Matplotlib

---

### 4️⃣ **Lyrics Generation using LSTM**
**Objective**: Generate song lyrics using **LSTM-based Recurrent Neural Networks (RNN)**.

**Steps**:
- Data Preprocessing: Cleaning and Tokenization
- Text to Sequence Transformation
- LSTM Model Implementation
- Model Training and Prediction
- Generating Lyrics from Seed Text

**Tools & Libraries**: TensorFlow, Keras, NumPy, Pandas, Matplotlib

---

### 5️⃣ **Mercedes-Benz Greener Manufacturing**
**Objective**: Reduce Mercedes-Benz manufacturing process runtime by predicting the target variable `y` using regression models.

**Steps**:
- Dimensionality Reduction with **PCA** (98% Variance Retention)
- Regression Models:
  - **XGBoost Regressor**
  - **Random Forest Regressor**
- Model Comparison and Performance Evaluation (MSE)

**Tools & Libraries**: Pandas, NumPy, Scikit-learn, XGBoost

---

### 6️⃣ **Movielens Case Study**
**Objective**: Analyze user ratings, identify top movies, and predict movie ratings using machine learning models.

**Steps**:
- Merging and Cleaning User, Movie, and Ratings Data
- Exploratory Data Analysis:
  - User Age Distribution
  - Top 25 Movies by Viewership
- Feature Engineering:
  - One-Hot Encoding for Genres
- Models for Rating Prediction:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Model Performance Evaluation

**Tools & Libraries**: Pandas, Seaborn, Matplotlib, Scikit-learn

---

### 7️⃣ **Sentiment Detection on IMDB Dataset**
**Objective**: Perform sentiment analysis on IMDB movie reviews using **Logistic Regression**, **Naive Bayes**, and **Decision Tree** models.

**Steps**:
- Text Preprocessing:
  - Removing HTML Tags, Punctuation, and Stopwords
  - Lemmatization
- Vectorization:
  - Bag-of-Words (BoW)
  - TF-IDF
- Hyperparameter Tuning:
  - **Logistic Regression**: GridSearchCV
  - **Naive Bayes**: Hyperparameter Optimization
  - **Decision Tree**: GridSearchCV
- Model Evaluation:
  - Accuracy, Precision, Recall, and F1-Score
  - Confusion Matrix for Results Visualization

| **Model**                | **Accuracy (BoW)** | **Accuracy (TF-IDF)** |
|--------------------------|--------------------|-----------------------|
| Logistic Regression      | 89.13%             | 88.95%                |
| Naive Bayes              | 85.73%             | 85.11%                |
| Decision Tree Classifier | 73.40%             | 73.40%                |

**Tools & Libraries**: Pandas, NLTK, NumPy, Scikit-learn, Seaborn, Matplotlib

---

## 🛠️ **Technologies Used**
- **Python**: TensorFlow, Keras, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, NLTK, XGBoost

---

## 📂 **Repository File Structure**
```plaintext
.
|-- Census_Solution.ipynb                     # Clustering on Census Dataset
|-- Deep Learning using Fashion MNIST.ipynb   # Fashion MNIST Classification
|-- Lending Club Loan Data Analysis.ipynb     # Loan Status Prediction
|-- Lyrics_Generation.ipynb                   # LSTM-based Lyrics Generation
|-- Mercedes-Benz Greener Manufacturing.ipynb # Regression for Runtime Prediction
|-- Movielens Case Study .ipynb               # Movie Ratings Analysis & Prediction
|-- Sentiment_Detection_imdb.ipynb            # Sentiment Analysis on IMDB Dataset
|-- README.md                                 # Repository Documentation
```

---

## ⚙️ **How to Run**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your_username/ml-dl-projects.git
   cd ml-dl-projects
   ```

2. **Install Dependencies**:
   ```bash
   pip install tensorflow keras pandas numpy seaborn matplotlib scikit-learn nltk xgboost
   ```

3. **Run the Notebooks**:
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open the respective project file (e.g., `Sentiment_Detection_imdb.ipynb`) and execute the cells.

---

## 🔮 **Future Enhancements**
- Implement **Transformer Models** (e.g., BERT) for sentiment analysis.
- Use **Collaborative Filtering** for improved movie recommendations.
- Optimize LSTM performance for longer lyrics generation.
- Add **Ensemble Methods** to combine models for better predictions.

---

## 👨‍💻 **Contributors**
- **Harmanan Kohli** - Data Scientist and ML/DL Enthusiast

---

## 📄 **License**
This repository is licensed under the MIT License.

---
