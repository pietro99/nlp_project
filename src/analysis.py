import matplotlib.pyplot as plt
import numpy as np
import json


def performance():
    # line 1 points
    LR_Y = [0.560,  0.678,0.7083, 0.6833, 0.6805, 0.6856, 0.6822, 0.6814, 0.6845 ]
    LR_X = [100, 600, 1200, 1800, 2000, 3000, 4000, 5000, 6000]
    # plotting the line 1 points 
    plt.plot(LR_X, LR_Y, label = "Linear Regression")
    # line 2 points
    DTC_Y = [0.580, 0.623, 0.6275, 0.6166, 0.6300, 0.6440, 0.6330, 0.6256, 0.6263]
    DTC_X = LR_X
    # plotting the line 2 points 
    plt.plot(DTC_X, DTC_Y, label = "Decision Tree")
    MLP_Y = [0.560, 0.5816,0.6132, 0.6127, 0.6295, 0.6360, 0.65675, 0.6258, 0.6396]
    MLP_X = LR_X 
    plt.plot(MLP_X, MLP_Y, label = "Multy-Layer Perceptron")
    plt.xlabel('dataset size')
    # Set the y axis label of the current axis.
    plt.ylabel('accuracy')
    # Set a title of the current axes.
    plt.title('performance comparison')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()

def processing():
    
    labels = ['Logistic Regression', 'Decision Tree', 'Multy-Layer Perceptron']
    men_means = [0.660, 0.605, 0.565]
    women_means = [0.665, 0.590, 0.52]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, men_means, width, label='No Numbers')
    rects2 = ax.bar(x + width/2, women_means, width, label='Numbers')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('accuracy')
    ax.set_title('influence of removing numbers per method')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


   
    fig.tight_layout()

    plt.show()
def common_words():
    with open("../data/vocab.json") as vocab_file:
        vocab = json.load(vocab_file)
    #print(vocab)
    sort_vocab = sorted(vocab.items(), key=lambda x: x[1])
    print(sort_vocab[-21:-1])
    dist = {}
    for i in sort_vocab:
        key = i[1]
        if key in dist:
            dist[key] += 1
        else:
            dist[key] = 1
   # print(dist)
    X = []
    Y = []
    for key in dist:
        if key<= 50:
            X.append(key)
            Y.append(dist[key])
    plt.plot(X, Y)
    plt.xlabel("top 50 words")
    plt.ylabel("word count")
    plt.title("vocabulary words distribution")

    plt.show()
processing()