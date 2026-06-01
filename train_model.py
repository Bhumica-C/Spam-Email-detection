import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load dataset
data = pd.read_csv("SMSSpamCollection", sep='\t', names=["label", "message"])

# Convert labels
data['label'] = data['label'].map({'ham':0, 'spam':1})

X = data['message']
y = data['label']

# Text vectorization
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

# SVM Model
model = SVC(probability=True)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("spam_model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("Model trained successfully")