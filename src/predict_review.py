from process import tokanize, clean, getVocabulary, getBOW
import json
from joblib import load
from sklearn.linear_model import LogisticRegression

print("write your review:")
review = [input()]

token = tokanize(review)
token = clean(token)
#print(token)
with open("../data/vocab.json") as vocab_file:
    vocab = json.load(vocab_file)
vector = getBOW(token, vocab)
clf = load("../models/BOW_LR_WEBSITE.joblib")
#print(vector)
prediction = clf.predict(vector)
print("your movie score is: "+str(prediction[0]))
