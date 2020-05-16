import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import math
from nltk.corpus import wordnet as wn


STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def computeSimilarity(first, second):
    """ritorna la similarità tra due liste di token"""
    return len(set(first).intersection(set(second)))/min(len(first),len(second))

def extract_sense_context(sense):
    """ Return a list of words extracted from examples,hyponim and hyperonims """

    examples = set()
    for sentence in sense.examples():
        examples.update(word_tokenize(sentence))
    
    for hyp in sense.hyponyms():
        for lemma in hyp.lemma_names():
            examples.update(lemma)
    
    for hyp in sense.hypernyms():
        for lemma in hyp.lemma_names():
            examples.update(lemma)

    examples.update(word_tokenize(sense.definition()))
    filtered_words = [LEMMATIZER.lemmatize(w.lower()) for w in examples if (not w in STOP_WORDS) and w.isalpha()]
    return filtered_words

def findBestSynset(words_aggregated_dict):
    """
    It takes in input a frequency word dict and return the best
    synset that it finds on wordnet
    """
    #1. Per ogni coppia di parole trovo il lowest common iperonimo
    #1.1 costruisco così un dizionario che ha: <iperonimo,punteggio> dove il punteggio è calcolato sommando la frequenza delle due parole che lo hanno trovato-radice(distanzaiperonimo)
        #se diverse coppie di parole danno lo stesso iperonimo allora il punteggio si somma
   
    hypers_score = {}

    for word, frequency in words_aggregated_dict.items():
        for word2, frequency2 in words_aggregated_dict.items():
            if word == word2:
                continue
            synset_word1 = wn.synsets(word)
            synset_word2 = wn.synsets(word2)
            if len(synset_word1) == 0 or len(synset_word2) == 0:
                continue
            synset_word1 = synset_word1[0]
            synset_word2 = synset_word2[0]

            lowest_common = synset_word1.lowest_common_hypernyms(synset_word2, simulate_root=False)
            if lowest_common is None or len(lowest_common)<=0:
                continue
            else:
                #print("---",lowest_common)
                lowest_common = lowest_common[0]
                hyper_distance = max(synset_word1.shortest_path_distance(lowest_common, simulate_root=True), synset_word2.shortest_path_distance(lowest_common, simulate_root=True))
                hyper_score = frequency+frequency2-hyper_distance
                
                #hyper_name = str(lowest_common.lemmas()[0])
                hyper_name = lowest_common
                #print("lowest common: ",hyper_name)

                #TODO:verificare se la in funziona bene
                if hyper_name in hypers_score:
                    hypers_score[hyper_name] += hyper_score
                else:
                    hypers_score[hyper_name] = hyper_score
    
    sorted_hypers_list = sorted(hypers_score.items(), key=lambda kv: kv[1], reverse=True)
    print(sorted_hypers_list)

    print("============ FOUND BEST HYPERONIMS")

    #1.2 prendo quindi i primi 5 iperonimi con il punteggio più alto ed estraggo 5 liste di parole dai loro esempi e glosse
    #2.prendo il primo iponimo di ogni termine di input ed estraggo il contesto anche da questi
    #3.calcolo overlap di ogni contesto dei termini con ogni contesto degli iperonimi [oppure posso fare similarità media di ogni coppia di termine] e prendo l'iperonimo che ha similarità cumulativa più alta
    context1 = extract_sense_context(sorted_hypers_list[0][0])
    context2 = extract_sense_context(sorted_hypers_list[1][0])
    context3 = extract_sense_context(sorted_hypers_list[2][0])
    context4 = extract_sense_context(sorted_hypers_list[3][0])
    context5 = extract_sense_context(sorted_hypers_list[4][0]) 

    print("LEN CONTEXTS: ",len(context1),len(context2),len(context3),len(context4),len(context5))

    #2.prendo il primo iponimo di ogni termine di input ed estraggo il contesto anche da questi
    #3.calcolo overlap di ogni contesto dei termini con ogni contesto degli iperonimi [oppure posso fare similarità media di ogni coppia di termine] e prendo l'iperonimo che ha similarità cumulativa più alta

    similarity1, best_hyp1 = compareHyponimWithHyperonimContext(words_aggregated_dict,context1)
    similarity2, best_hyp2 = compareHyponimWithHyperonimContext(words_aggregated_dict,context2)
    similarity3, best_hyp3 = compareHyponimWithHyperonimContext(words_aggregated_dict,context3)
    similarity4, best_hyp4 = compareHyponimWithHyperonimContext(words_aggregated_dict,context4)
    similarity5, best_hyp5 = compareHyponimWithHyperonimContext(words_aggregated_dict,context5)

    print("@@@@@@@@@@@@@@@@@@@@@@@ ",best_hyp1,best_hyp2,best_hyp3,best_hyp4,best_hyp5)

    lst = [similarity1,similarity2,similarity3,similarity4,similarity5]
    best_context_pos = lst.index(max(lst))
    return sorted_hypers_list[best_context_pos]


def compareHyponimWithHyperonimContext(words_aggregated_dict,hyper_context):
    """
    Extracts first hyponim context from each term in words_aggregated
    and compute a similarity overlap between all hyponims contexts and the hyper_context.
    It returns the average similarity.
    """
    similarity = 0
    count = 0
    best_hyponim = None #best synset that is an hyponim of a words_aggregated
    max_sim = 0
    for word, frequency in words_aggregated_dict.items():
        synset_word = wn.synsets(word)
        if len(synset_word) == 0:
            continue
        synset_word_hyponim = synset_word[0].hyponyms()
        if len(synset_word_hyponim) == 0:
            continue
        synset_word_hyponim = synset_word_hyponim[0]
        hyponim_context = extract_sense_context(synset_word_hyponim)
        similarity_this = computeSimilarity(hyponim_context,hyper_context)
        similarity += similarity_this
        if similarity_this > max_sim:
            max_sim = similarity_this
            best_hyponim = synset_word_hyponim
        count+=1
    return similarity/count, best_hyponim

def main():
    #read csv and transponse it
    df = pd.read_csv('contentToForm.csv').T

    #remove stopwords
    concept_list = [] #it is a list of token list lemmatized and cleaned
    df = df.dropna()
    df = df.apply(lambda serie: [word_tokenize(word) for word in serie.tolist()])
    all_concepts = []
    for column in df.columns:
        concept_list = []
        for value_list in df[column].tolist():
            concept_list.append([LEMMATIZER.lemmatize(w.lower()) for w in value_list if (isinstance(w,str)) and (not w in STOP_WORDS) and w.isalpha()])
        all_concepts.append(concept_list)
    
    
    #ciclo per concetto
    i = 0
    for concept_tokens in all_concepts:
        #concept_tokens is a list of list of tokens. Each sublist is a single definition
        words_aggregated_dict = defaultdict(int)
        for sublist in concept_tokens:
            for word in sublist:
                if word == 'concetto':
                    continue
                words_aggregated_dict[word] += 1

        bestSynset = findBestSynset(words_aggregated_dict)
        print("Best synset for concept ",i,"= ",bestSynset)
        i+=1


if __name__ == "__main__":
    main()