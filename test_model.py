import joblib
import json
import os

def run_tests(model_path="toxic_model_latest.hush", vectorizer_path="vectorizer_latest.hush"):
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("Error: Model or Vectorizer files missing. Run trainer.py first.")
        return

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Comprehensive Test Cases
    test_cases = [
        # Non-Toxic
        ("I'm going to the store to buy some apples and bread.", 0),
        ("The presentation you gave today was very informative, thank you.", 0),
        ("It's a beautiful day outside, let's go for a walk in the park.", 0),
        ("You are a Jew",0),
        ("You are a Christian",0),
        ("You are a Muslim",0),
        ("You ae a Buddhist",0),
        ("I am a Jew",0),
        ("I am a Christian",0),
        ("I am a Muslim",0),
        ("I am a Buddhist",0),
        ("what are you doing",0),
        ("how are you",0),
        ("what is your name",0),
        ("where are you from",0),

        
        # Obvious Toxic
        ("You are a total idiot and I hate you.", 1),
        ("I will come to your house and destroy everything you love.", 1),
        
        # Subtle Toxicity / Gaslighting
        ("It's so brave of you to go out in public looking like that.", 1),
        ("I'm sure you did your best, even if your best is objectively pathetic.", 1),
        ("I don't expect you to understand this, given your limited mental capacity.", 1),
        
        # Late-Text Insults (Long Form)
        ("I really appreciated the effort you put into the project plan. The milestones were clear, the resource allocation seemed fair, and the timeline was realistic. It's just a shame that your personality is as toxic as radioactive waste.", 1),
        ("The dinner last night was wonderful. The food was cooked perfectly, the wine selection was excellent, and the company was great. However, looking at your face throughout the meal made me want to vomit.", 1),
        
        # Potential False Positives (Edge Cases)
        ("This code is killing me, I need to debug it.", 0),
        ("That movie was so bad it was almost funny.", 0),
        ("He is an ass for leaving without saying goodbye, but I still like him.", 0)
    ]

    print(f"{'PRED':<10} | {'EXPECT':<10} | {'TEXT'}")
    print("-" * 80)

    correct = 0
    for text, expected in test_cases:
        vec = vectorizer.transform([text])
        # Use predict_proba if available for confidence
        pred = model.predict(vec)[0]
        
        status = "PASS" if pred == expected else "FAIL"
        label_pred = "TOXIC" if pred == 1 else "CLEAN"
        label_exp = "TOXIC" if expected == 1 else "CLEAN"
        
        print(f"{label_pred:<10} | {label_exp:<10} | {text[:60]}...")
        if pred == expected:
            correct += 1

    print("-" * 80)
    print(f"Test Results: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.2f}%)")

if __name__ == "__main__":
    run_tests()