import pandas as pd
import json
import os
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model(data_path="classification_data.csv", version=None):
    # Set version based on timestamp if not provided
    if version is None:
        version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"--- Training Version: {version} ---")
    
    # Load and clean data
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = pd.read_csv(data_path).dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )

    # Configure Vectorizer
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5), 
        max_features=50000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Model
    model = SGDClassifier(loss='modified_huber', penalty='l2', alpha=0.0001, random_state=42, class_weight='balanced')
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")

    # Prepare Artifacts Paths
    model_name = f"toxic_model_v{version}.hush"
    vec_name = f"vectorizer_v{version}.hush"
    meta_name = f"metadata_v{version}.json"

    # Save Artifacts
    joblib.dump(model, model_name)
    joblib.dump(vectorizer, vec_name)
    
    # Create Latest Symlinks (or just copies for simplicity)
    joblib.dump(model, 'toxic_model_latest.hush')
    joblib.dump(vectorizer, 'vectorizer_latest.hush')

    # Save Metadata
    metadata = {
        "version": version,
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset_source": data_path,
        "metrics": {
            "accuracy": acc,
            "precision_toxic": report['1']['precision'],
            "recall_toxic": report['1']['recall']
        },
        "params": {
            "ngram_range": [1, 3],
            "max_features": 10000,
            "loss": "modified_huber"
        }
    }

    with open(meta_name, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Successfully saved artifacts: {model_name}, {vec_name}, {meta_name}")
    return version

if __name__ == "__main__":
    train_model()