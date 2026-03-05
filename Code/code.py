import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# Load dataset
fake_news = pd.read_csv('dataset/fake.csv')
true_news = pd.read_csv('dataset/true.csv')

fake_news['label'] = 0
true_news['label'] = 1

news = pd.concat([fake_news, true_news], ignore_index=True)

# Fill null values
news = news.fillna('')

# Merge title and date
news['content'] = news['title'] + " " + news['date']

# Stemming
port_stem = PorterStemmer()

def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [port_stem.stem(word) for word in content if word not in stopwords.words('english')]
    return " ".join(content)

news['content'] = news['content'].apply(stemming)

# Separate data and labels
X = news['content'].values
Y = news['label'].values

# Convert text to numerical values
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy
train_pred = model.predict(X_train)
train_acc = accuracy_score(train_pred, Y_train)

test_pred = model.predict(X_test)
test_acc = accuracy_score(test_pred, Y_test)

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

# Prediction example
sample = X_test[9]
sample = sample.reshape(1, -1)

prediction = model.predict(sample)

if prediction[0] == 0:
    print("The news is Fake")
else:
    print("The news is True")