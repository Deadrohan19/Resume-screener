import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the data
df = pd.read_csv('UpdatedResumeDataSet.csv')
print(df.head())
print("Dataset shape:", df.shape)

# Check category distribution
print("Category distribution:\n", df['Category'].value_counts())

# Visualize category distribution with count plot
plt.figure(figsize=(15,5))
sns.countplot(df['Category'])
plt.xticks(rotation=90)
plt.show()

# Pie chart for category distribution
counts = df['Category'].value_counts()
labels = df['Category'].unique()
plt.figure(figsize=(15,10))
plt.pie(counts, labels=labels, autopct='%1.1f%%', shadow=True, colors=plt.cm.plasma(np.linspace(0,1,3)))
plt.show()

# Define resume cleaning function
def cleanResume(resume_text):
    clean_text = re.sub(r'http\S+\s*', ' ', resume_text)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+', '', clean_text)
    clean_text = re.sub(r'@\S+', '  ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', clean_text)  # Removed extra backslash
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text

# Apply cleaning function to resume text
df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

# Encode categories as integers
le = LabelEncoder()
le.fit(df['Category'])
df['Category'] = le.transform(df['Category'])

# Convert resume text to TF-IDF features
tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
requiredText = tfidf.transform(df['Resume'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(requiredText, df['Category'], test_size=0.2, random_state=42)

# Train a K-Nearest Neighbors classifier with One-vs-Rest strategy for multi-class classification
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

# Make predictions and calculate accuracy
ypred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, ypred))

# Save the TF-IDF vectorizer and trained classifier
joblib.dump(tfidf, 'tfidf.joblib')
joblib.dump(clf, 'clf.joblib')
