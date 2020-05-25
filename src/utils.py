
import os
DATA_SIZE = 100
def readTrainingData():
    sentences = []
    scores = []
    counter = 0
    for filename in os.listdir('../data/train/pos'):
        counter +=1
        if filename.endswith('.txt'):
            with open(os.path.join('../data/train/pos', filename), encoding="utf8") as f:
                sentences.append(f.read())
                score = filename.split(".")[0]
                score = score.split("_")[1]
                scores.append(int(score))
        if counter>=DATA_SIZE:
            print("1/4")
            break
    counter = 0
    for filename in os.listdir('../data/train/neg'):
        counter+=1
        if filename.endswith('.txt'):
            with open(os.path.join('../data/train/neg', filename), encoding="utf8") as f:
                sentences.append(f.read())
                score = filename.split(".")[0]
                score = score.split("_")[1]
                scores.append(int(score))
        if counter>=DATA_SIZE:
            print("2/4")
            break
    return sentences, scores

def readTestingData():
    sentences = []
    scores = []
    counter = 0
    for filename in os.listdir('../data/train/pos'):
        counter+=1
        # print(filename)
        if filename.endswith('.txt'):
            with open(os.path.join('../data/train/pos', filename), encoding="utf8") as f:
                sentences.append(f.read())
                score = filename.split(".")[0]
                score = score.split("_")[1]
                scores.append(int(score))
        if counter>=DATA_SIZE:
            print("3/4")
            break
    counter = 0
    for filename in os.listdir('../data/test/neg'):
        counter+=1
        # print(filename)
        if filename.endswith('.txt'):
            with open(os.path.join('../data/test/neg', filename), encoding="utf8") as f:
                sentences.append(f.read())
                score = filename.split(".")[0]
                score = score.split("_")[1]
                scores.append(int(score))
        if counter>=DATA_SIZE:
            print("4/4")
            break
    return sentences, scores


