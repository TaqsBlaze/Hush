import joblib
import sys



model = joblib.load("toxic_model.hush")
vectorizer = joblib.load("vectorizer.hush")



new_texts = """This is a normal text
this big idiot must be killed
I fucked his girl friend
such an idiot
this man is stupid
this guy is good but stupid
I will kill him
he will die
fuck off
I went out last night with a friend
Last night I met a guy and he was such an ass
welcome to our new offices
I had sex last night
he is having sex
some people do not deserve to be here
Last night I want out with a guy and he was such an ass
I dont think you and me will ever go out again I dont like you at all""".split("\n")

new_vec = vectorizer.transform(new_texts)
prediction = model.predict(new_vec)

for text, pred in zip(new_texts, prediction):
    label = "Toxic" if pred == 1 else "Non-Toxic"
    print(f"Text: {text} ---> Prediction: {label}")
