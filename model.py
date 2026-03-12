import joblib
import sys



model = joblib.load("toxic_model_latest.hush")
vectorizer = joblib.load("vectorizer_latest.hush")


try:
    text = [sys.argv[1]]


except Exception as error:

    text = str(input("Enter message: "))

    text = [text]


vec = vectorizer.transform(text)

prediction = model.predict(vec)

print(prediction)

if prediction[0] > 0:
    print("Your message is toxic and will not be sent")
else:
    print("Your message is clean and was sent")
