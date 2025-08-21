import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                 sep='\t', header=None, names=['label', 'message'])
print("First 5 rows of the dataset:")
print(df.head())
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sample_msg = ["Congratulations! You have won a free ticket to Bahamas. Call now!", 
              "Hi friend, let's meet tomorrow at 5 PM."]
sample_tfidf = vectorizer.transform(sample_msg)
predictions = model.predict(sample_tfidf)

for msg, pred in zip(sample_msg, predictions):
    print(f"\nMessage: {msg}")
    print("Prediction:", "Spam" if pred == 1 else "Ham")