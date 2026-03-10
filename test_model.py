import joblib
import sys



model = joblib.load("toxic_model.md5")
vectorizer = joblib.load("vectorizer.md5")



new_texts = """This is a normal text
this big idiot must be killed
I fucked his girl friend
such an idiot
this man is stupid
this guy is good but stupid
I will kill him
he will die
fuck off""".split("\n")

new_vec = vectorizer.transform(new_texts)
prediction = model.predict(new_vec)

for text, pred in zip(new_texts, prediction):
    label = "Toxic" if pred == 1 else "Non-Toxic"
    print(f"Text: {text} ---> Prediction: {label}")
