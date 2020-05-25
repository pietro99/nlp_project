from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk



def tokanize(sentences):
    tokenList = [[]]
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tokenList.append(tokens)
    tokenList.remove(tokenList[0])
    return tokenList

def clean(tokens):
    cleanedTokens = [[]]
    stop_words = stopwords.words('english')
    stop_words.append("br")
    porter = PorterStemmer()
    for token in tokens:
        # to lower case:
        token = [word.lower() for word in token]
        #remove non alphabetic tokens
       # token = [word for word in token if word.isalpha()]
        #filter out stop words
        token = [word for word in token if not word in stop_words]
        # stemming
        token = [porter.stem(word) for word in token]
        cleanedTokens.append(token)
    cleanedTokens.remove(cleanedTokens[0])
    return cleanedTokens

def getVocabulary(tokens):
    vocab = {}
    for sentence in tokens:
        for token in sentence:
            if token in vocab:
                vocab[token] += 1
            else:
                vocab[token] = 1
   
    return vocab

def getBOW(tokens, vocab):
    vectors = []
    for sentence in tokens:
        sentence_vector = []
        for token in vocab:
            if token in sentence:
                sentence_vector.append(1)
            else:
                sentence_vector.append(0)
        vectors.append(sentence_vector)
    return vectors



