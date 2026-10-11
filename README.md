Fake News Detection System

A machine learning model designed to classify news articles as REAL or FAKE using Natural Language Processing and linear classification algorithms.

Features
- Natural Language Processing: Uses TfidfVectorizer for text feature extraction.
- Machine Learning Model: Utilizes PassiveAggressiveClassifier for accurate classification.
- Robust Pipeline: Implements fallback logic to support both custom datasets and built-in demonstration samples.
- Custom Testing: Evaluates real-time input headlines.

Tech Stack
- Language: Python
- Libraries: Pandas, NumPy, Scikit-Learn

Installation & Usage

1. Clone the repository:
git clone https://github.com/Harshit-kuthar/fake-news-detection.git
cd fake-news-detection

2. Install dependencies:
pip install pandas numpy scikit-learn

3. Run the script:
python fake_news.py

Author
Harshit 
