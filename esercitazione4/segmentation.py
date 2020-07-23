import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import math
from nltk.corpus import wordnet as wn
from itertools import product

#source: https://textfiles.vistech.net/news/

MAX_ITERATIONS = 10
INITIAL_WINDOWS = 2
MIN_COHESION_IN_WINDOW = 0.9

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def readDoc():
    """
    Read the doc and parse it in sentences and tokens
    """
    with open('C:\\Workspaces\\TLN_DiCaro\\esercitazione4\\testo.txt', 'r') as myfile:
        data = myfile.read()
    sentences = data.split("\n")

    #remove empty lines
    sentences = [k for k in sentences if k and k!="\n"]

    #tokenize each sentence
    tokens = []
    for sentence in sentences:
        tokens_sentence = word_tokenize(sentence)
        filtered_tokens = [LEMMATIZER.lemmatize(w.lower()) for w in tokens_sentence if (not w in STOP_WORDS) and w.isalpha()]
        tokens.append(filtered_tokens)
    return sentences, tokens


def calculateWindows(sentences,initial_windows_size):
    """
    It split the sentences list into sublist based on initial_windows_size
    """
    #print(sentences,initial_windows_size )
    return [sentences[i * initial_windows_size:(i + 1) * initial_windows_size] for i in range((len(sentences) + initial_windows_size - 1) // initial_windows_size )]  

def calculateCohesion(window):
    """
    It returns the cohesion average of the window sentences using wup_similarity
    calculating it for each sequential couple.
    It return also the couple with less cohesion
    """
    if len(window) == 1:
        #se c'è solo una frase in questa finestra allora la coesione è massima
        return (1,None)

    index1 = 0
    index2 = 1
    avg_similarity = 0
    #breakpoint = i --> break between i and i+1
    breakpoint = 0
    min_similarity = 100000000
    while index2 < len(window):
        #calcolo tutti i possibili synset per le parole delle frasi window[index1] e window[index2]
        allsyns1 = set(ss for word in window[index1] for ss in wn.synsets(word))
        allsyns2 = set(ss for word in window[index2] for ss in wn.synsets(word))
        if len(allsyns1) > 0 and len(allsyns2)>0:   
            #calcolo max wup similarity tra ogni coppia di synset  
            best, syn1, syn2 = max((wn.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))

            avg_similarity += best
            if min_similarity > best:
                min_similarity = best
                breakpoint = index1

        index1+=1
        index2+=1

    avg_similarity = avg_similarity/len(window)
    return avg_similarity, breakpoint


def splitWindow(newWindows,best_breakpoint_window, windowIndex):
    """
    It removes newWindows[windowIndex] and split it in two
    using best_breakpoint_window
    """
    window = newWindows.pop(windowIndex)
    if len(window) < 2:
        return newWindows
    newWindow1 = []
    newWindow2 = []
    index = 0
    while index < len(window):
        if index <= best_breakpoint_window:
            newWindow1.append(window[index])
        else:
            newWindow2.append(window[index])
        index+=1
    
    newWindows.insert(windowIndex, newWindow2)
    newWindows.insert(windowIndex, newWindow1)
    return newWindows

def main():
    #leggo documento e splitto in frasi e token
    sentences, tokens = readDoc()
    print("There are ",len(sentences), "sentences")


    #inizialmente splitto le finestre mettendo lo stesso numero di frasi
    initial_windows_size = len(sentences)//INITIAL_WINDOWS
    if initial_windows_size > 7:
        initial_windows_size -= 5
    
    windows = calculateWindows(tokens,initial_windows_size)
    filtered_windows = []
    for window in windows:
        filtered_windows.append([x for x in window if x != []])
    windows = filtered_windows
    print("INITIAL WINDOWS")
    baseline = 0
    for initial_window in windows:
        #print("len:",len(initial_window))
        print(len(initial_window)+baseline)
        baseline += len(initial_window)

    ### INIZIO A ITERARE ###

    iteration = 0
    changes = 1
    newWindows = windows
    actual_cohesion_threshold = MIN_COHESION_IN_WINDOW
    while iteration < MAX_ITERATIONS and changes > 0:
        print("@@@@@@@@@@@@@ ITERATION NUMBER ",iteration," @@@@@@@@@@@@@")
        changes = 0
        index = 0
        for window in windows:
            #calcolo coesione media di ogni coppia di frasi e il miglior breakpoint(la coppia con coesione più bassa)
            cohesion_avg, best_breakpoint = calculateCohesion(window)
            #print("cohesion avg: ",cohesion_avg)
            if cohesion_avg < actual_cohesion_threshold and len(window) > 2:
                print("Found break point at window ",index," at sentence",best_breakpoint," with an avg cohesion of ",cohesion_avg)
                changes+=1
                actual_cohesion_threshold = actual_cohesion_threshold*0.96 #diminuisco la coesione minima all'aumentare delle window
                #print("LEN BEFORE: ",len(newWindows))
                #splitta in due la window
                newWindows = splitWindow(newWindows,best_breakpoint, index)
                
                #print("LEN AFTER: ",len(newWindows))
            index+=1
        windows = newWindows
        iteration +=1

    baseline = 0
    print("Converged after ", iteration, "iterations")
    for final_window in newWindows:
        print("===================================== Finish at sentence row: ", len(final_window)+baseline)
        print(final_window)
        print("===================================== SIZE: ",len(final_window),"\n")
        baseline += len(final_window)



if __name__ == "__main__":
    main()