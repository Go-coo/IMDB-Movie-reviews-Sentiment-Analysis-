# IMDB-Movie-reviews-Sentiment-Analysis-

## 📌 Project Overview
This project performs **sentiment analysis** on IMDb movie reviews using **Natural Language Processing (NLP)** and **Machine Learning**. It classifies reviews as **positive** or **negative** using a **Logistic Regression model** trained on a preprocessed dataset.

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- NLTK (Natural Language Toolkit)
- Scikit-learn (Machine Learning)
- TF-IDF Vectorization

## 📂 Dataset
The dataset used is the **IMDb Reviews Dataset**, which contains movie reviews labeled as **positive** or **negative**.

## 🔧 Installation & Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-imdb.git
   cd sentiment-analysis-imdb
   ```
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn nltk
   ```
3. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## 🚀 How to Run
1. Open the **Jupyter Notebook** or run the script.
2. Load the dataset (`IMDB Dataset.csv`).
3. Train the model using `LogisticRegression`.
4. Test with new custom reviews:
   ```python
   print(predict_sentiment(preprocess_text("I loved this movie!")))
   ```

## 📊 Model Performance
- **Accuracy:** Achieved around **89%** on the test set.

## 📜 Features Implemented
✔ Data Preprocessing (Stopwords removal, Tokenization, TF-IDF)
✔ Sentiment Classification (Positive/Negative)
✔ Custom Review Prediction
✔ Model Performance Evaluation

## 📌 Next Steps
🔹 Improve accuracy using **Deep Learning (LSTMs, Transformers)**
🔹 Deploy the model as a **Web API** for real-time predictions

## 💡 Author
Developed by **[GOKUL M]**

If you found this helpful, feel free to ⭐ the repo!

