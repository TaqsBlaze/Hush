![icon](https://raw.githubusercontent.com/TaqsBlaze/Hush/refs/heads/main/image/image.png)

# Hush - Text Classifier

Hush is a lightweight, efficient machine learning model designed to detect toxic language in text. Using a combination of Natural Language Processing (NLP) techniques and ensemble learning, Hush classifies phrases into "Toxic" (1) and "Non-Toxic" (0) categories, helping to maintain healthier digital environments by "hushing" harmful content.

## 🚀 Features

Fast Inference: Built using Scikit-Learn for rapid classification.

Context-Aware Vectorization: Utilizes TfidfVectorizer with n-gram support (bigrams) and English stop-word filtering to better understand context and ignore noise.

Regularized Performance: Uses a RandomForestClassifier with constrained depth and leaf nodes to prevent overfitting and ensure better generalization on unseen data.

## 📂 Project Structure

classification_data.csv: The training dataset containing labeled text samples.

[trainer.py](trainer.py): Script to train the model, evaluate performance, and export the binary artifacts.

[test_model.py](test_model.py): Batch testing script to run predictions on a predefined list of text samples.

[model.py](model.py): Interactive inference script for real-time message testing.

[toxic_model.md](toxic_model.md): The saved Random Forest model (generated after training).

[vectorizer.md](vectorizer.md): The saved TF-IDF vectorizer (generated after training).


## 🛠️ Installation & Setup

Clone the repository (or ensure all project files are in one directory).

Install dependencies:

pip install pandas scikit-learn joblib


## 🏋️ Training the Model

To train the model from scratch using the provided dataset, run the trainer script:

python trainer.py


This will:

Load, clean, and shuffle the dataset for a balanced 80/20 train-test split.

Fit the TF-IDF vectorizer (including unigrams and bigrams).

Train the Random Forest classifier with regularization parameters.

Output a classification report (Precision, Recall, F1-Score).

Save the model and vectorizer as .md files.

## 🧪 Testing and Inference

Batch Testing

To run a batch test against a list of hardcoded samples:

python test_model.py


Interactive Single Message Testing

Use model.py to test individual messages. You can provide the message as a command-line argument or enter it manually when prompted:

Option A: Command Line Argument

python model.py "You are doing a great job"


Option B: Interactive Prompt

python model.py
# The script will prompt: Enter message: 


Example Usage Logic:

import joblib

### Load artifacts
model = joblib.load("toxic_model.hush")
vectorizer = joblib.load("vectorizer.hush")

# Predict using Hush logic
text = ["Please be kind to others"]
vec = vectorizer.transform(text)
prediction = model.predict(vec)

print("Toxic" if prediction[0] == 1 else "Non-Toxic")


## 📊 Technical Details

Algorithm: Random Forest Classifier

n_estimators=100

max_depth=20

min_samples_leaf=2

class_weight='balanced'

Feature Extraction: TF-IDF Vectorization

ngram_range=(1, 2)

stop_words='english'

max_features=2000

Data Split: 80% Train / 20% Test (Shuffled)

## 📝 License

[MIT](LICENSE)

This project is open-source and available for educational and moderation purposes.