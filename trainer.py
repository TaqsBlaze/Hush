import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv("classification_data.csv")

# Clean data: drop any rows with missing values that might have occurred during expansion
df = df.dropna()


X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42, 
    shuffle=True
)

vectorizer = TfidfVectorizer(
    stop_words='english', 
    ngram_range=(1, 2), 
    max_features=2000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model with regularization:
# 1. max_depth=20 prevents the trees from becoming too deep/complex.
# 2. min_samples_leaf=2 ensures nodes only split if they result in meaningful groups.
# 3. class_weight='balanced' helps if there's any slight imbalance in labels.
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=20, 
    min_samples_leaf=2,
    random_state=21,
    class_weight='balanced'
)

model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Classification Report (Optimized for Overfitting):")
print(classification_report(y_test, y_pred))


joblib.dump(model, 'toxic_model.md')
joblib.dump(vectorizer, 'vectorizer.md')

print("Model and Vectorizer saved successfully.")