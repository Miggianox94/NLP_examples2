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

''' ############ word disambiguation functions ###############'''

def leskAlgoHyperonims(synset,contextList):
    """
    Ritorna l'iperonimo più simile al contesto dato in input
    """
    besthyper = None
    max_overlap = -1
    for hyper in synset.hypernyms():
        #print("POS HYP= ",hyponim.pos())
        if hyper.pos() != 'n':
            continue
        signature = extract_sense_context(hyper, True)
        overlap = computeoverlap(signature,contextList)
        if overlap > max_overlap:
            max_overlap = overlap
            besthyper = hyper
    return besthyper

def leskAlgoHyponims(word,sentence):
    """
    Ritorna l'iponimo più simile al più simile wordsense data la parola e la sentence che funge da contesto di disambiguazione
    """
    best_sense = None
    context = computecontext(sentence)
    max_overlap = -1
    for sense in wn.synsets(word):
        signature = extract_sense_context(sense)
        overlap = computeoverlap(signature,context)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense

    if best_sense is None:
        return None
    
    max_overlap = -1
    besthyponim = None
    for hyponim in best_sense.hyponyms():
        #print("POS HYP= ",hyponim.pos())
        if hyponim.pos() != 'n':
            continue
        signature = extract_sense_context(hyponim)
        overlap = computeoverlap(signature,context)
        if overlap > max_overlap:
            max_overlap = overlap
            besthyponim = hyponim
    return besthyponim

def leskAlgo(word,sentence):
    best_sense = None
    context = computecontext(sentence)
    max_overlap = -1
    for sense in wn.synsets(word):
        signature = extract_sense_context(sense)
        overlap = computeoverlap(signature,context)
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    return best_sense

def computeoverlap(signature, context):
    return len(set(signature).intersection(set(context)))

def computecontext(sentence):
    filtered_sentence = [LEMMATIZER.lemmatize(w.lower()) for w in sentence if (not w in STOP_WORDS) and w.isalpha()]
    return filtered_sentence

''' ########################################## '''

def extract_sense_context(sense, no_hyper = False):
    """ Return a list of words extracted from examples,hyponim and hyperonims """

    examples = set()
    for sentence in sense.examples():
        examples.update(word_tokenize(sentence))
    
    for hyp in sense.hyponyms():
        for lemma in hyp.lemma_names():
            examples.update(lemma)

    if no_hyper:
        filtered_words = [LEMMATIZER.lemmatize(w.lower()) for w in examples if (not w in STOP_WORDS) and w.isalpha()]
        return filtered_words
    
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
    #trovo contesto
    context = list(words_aggregated_dict.keys())
    context_original = context.copy()
    word_synsets = {}
    #print("DICT: ",words_aggregated_dict.items())
    for word, frequency in words_aggregated_dict.items():
        synset_word = leskAlgo(word,words_aggregated_dict.keys())
        if synset_word is None:
            #print("SKIPPED")
            continue
        #print("distance from root: ",min([len(path) for path in synset_word.hypernym_paths()]))
        #se il synset trovato è troppo "generico" allora lo scarto
        if min([len(path) for path in synset_word.hypernym_paths()]) < 8:
            continue
        word_synsets[synset_word] = frequency
        context_original.extend(extract_sense_context(synset_word, True))



    #genero una lista con tutti gli iponimi di ogni iperonimo trovato di ogni synset
    hyponims_candidates = []
    for synset, frequency in word_synsets.items():
        if frequency < 3:
            #se la frequenza del termine è nella "norma" allora cerco gli iponimi dei suoi iperonimi
            for hyper in synset.hypernyms():
                if min([len(path) for path in hyper.hypernym_paths()]) < 8:
                    continue
                for hypon in hyper.hyponyms():
                    if min([len(path) for path in hyper.hypernym_paths()]) < 8:
                        continue
                    if hypon.pos() != 'n':
                        continue
                    hyponims_candidates.append(hypon)
        else:
            #se la frequenza è alta allora uso lo stesso termine come iperonimo
            for hypon in synset.hyponyms():
                if hypon.pos() != 'n':
                    continue
                hyponims_candidates.append(hypon)
    print("Hyponim candidates: ",hyponims_candidates)




    #.calcolo overlap di ogni contesto dei termini con ogni contesto degli iponimi e prendo l'iponimo che ha similarità più alta
    best_hyponim = None
    best_sim = -1
    for hyponim in hyponims_candidates:
        context_hyponim = extract_sense_context(hyponim, False)
        #similarity, maxSim = computeSimilarity(hyponim,word_synsets.keys())
        #similarity = similarity + maxSim/2
        similarity = computeoverlap(context,context_hyponim)#/max(len(context),len(context_hyponim))
        if similarity > best_sim:
            print("similarity change: ", similarity)
            best_sim = similarity
            best_hyponim = hyponim
    return best_hyponim
    

def computeSimilarity(synset,listSynset):
    """
    calculate average similarity using wupalmer
    """
    similarity = 0
    max_sym = 0
    for syn in listSynset:
        similarity_single = synset.wup_similarity(syn)
        if similarity_single is not None:
            similarity += similarity_single
            max_sym = max(max_sym,similarity_single)
    
    return similarity/len(listSynset),max_sym

def compareHyponimWithHyperonimContext(words_aggregated_dict,hyper_context):
    """
    Extracts best hyponim context from each term in words_aggregated
    and compute a similarity overlap between all hyponims contexts and the hyper_context.
    It returns the average similarity.
    """
    similarity = 0
    count = 0
    best_hyponim = None #best synset that is an hyponim of a words_aggregated
    max_sim = 0
    keysList = list(words_aggregated_dict.keys())
    #keysList.extend(hyper_context)
    for word, frequency in words_aggregated_dict.items():
        synset_word = wn.synsets(word)
        if len(synset_word) == 0:
            continue
        
        #word disambiguation
        synset_word_hyponim = leskAlgoHyponims(word,keysList)
        if synset_word_hyponim is None:
            continue
        hyponim_context = extract_sense_context(synset_word_hyponim)
        #print("Hyponim context len: ",len(hyponim_context))
        merged = []
        #merged.extend(hyper_context)
        merged.extend(keysList)
        similarity_this = computeSimilarity(hyponim_context,merged)
        similarity += similarity_this
        if similarity_this > max_sim:
            max_sim = similarity_this
            best_hyponim = synset_word_hyponim
            #print("best_hyponim CHANGE: ",best_hyponim," with similarity: ",similarity_this)
        count+=1
    #print(similarity,"-------",count)
    return similarity/count, best_hyponim

def main():
    #read csv and transponse it
    df = pd.read_csv('contentToForm.csv').T

    #remove stopwords
    concept_list = [] #it is a list of token list lemmatized and cleaned
    df = df.dropna()
    df = df.apply(lambda serie: [word_tokenize(word) for word in serie.tolist()])
    all_concepts = []
    print(df)
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

        besthyponim= findBestSynset(words_aggregated_dict)
        #print(words_aggregated_dict)
        print("\n===============================================================")
        print("========== Best synset for concept ",i,"= ",besthyponim)
        print("===============================================================\n")
        i+=1


if __name__ == "__main__":
    #I concetti corretti sono: politics, justice, greed, patience, food, vehicle, screw, radiator
    main()