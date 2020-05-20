from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import math
from nltk.corpus import wordnet as wn
import nltk

N_INSTANCES_TO_GET = 20000
VALENCE = 2
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


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

def extract_sense_context(sense, only_examples = False):
    """ Return a list of words extracted from examples,hyponim and hyperonims """

    examples = set()
    for sentence in sense.examples():
        examples.update(word_tokenize(sentence))
    
    if only_examples:
        filtered_words = [LEMMATIZER.lemmatize(w.lower()) for w in examples if (not w in STOP_WORDS) and w.isalpha()]
        return filtered_words
    
    for hyp in sense.hyponyms():
        for lemma in hyp.lemma_names():
            examples.update(lemma)
    
    for hyp in sense.hypernyms():
        for lemma in hyp.lemma_names():
            examples.update(lemma)

    examples.update(word_tokenize(sense.definition()))
    filtered_words = [LEMMATIZER.lemmatize(w.lower()) for w in examples if (not w in STOP_WORDS) and w.isalpha()]
    return filtered_words

def computeoverlap(signature, context):
    return len(set(signature).intersection(set(context)))

def computecontext(sentence):
    filtered_sentence = [LEMMATIZER.lemmatize(w.lower()) for w in sentence if (not w in STOP_WORDS) and w.isalpha()]
    return filtered_sentence

def findright(verb_index,sentence):
    """
    Return the wordnet lexname of the first pos
    on the right of the verb that is not a stop word
    """
    #print(sentence,verb_index)
    i = verb_index+1
    while i<len(sentence):
        if (not sentence[i][0] in STOP_WORDS) and sentence[i][0].isalpha():
            syn = leskAlgo(LEMMATIZER.lemmatize(sentence[i][0].lower()),computecontext([row[0] for row in sentence]))
            #synsets = wn.synsets(LEMMATIZER.lemmatize(sentence[i][0].lower()))
            if syn is None:
                return None
            return syn.lexname()
        i+=1
    return None

def findleft(verb_index,sentence):
    """
    Return the wordnet lexname of the first pos
    on the left of the verb that is not a stop word
    """
    i = verb_index-1
    while i>0:
        if (not sentence[i][0] in STOP_WORDS) and sentence[i][0].isalpha():
            syn = leskAlgo(LEMMATIZER.lemmatize(sentence[i][0].lower()),computecontext([row[0] for row in sentence]))
            #synsets = wn.synsets(LEMMATIZER.lemmatize(sentence[i][0].lower()))
            if syn is None:
                return None
            return syn.lexname()
        i-=1
    return None

def getFillers():
    """
    Return a list of N_INSTANCES_TO_GET of couples (left and right filler)
    that contains verb TO BUILD
    """
    count = 0
    couples = []
    print(len(brown.tagged_sents(tagset='universal')))
    total_processed = 0
    for sentence in brown.tagged_sents(tagset='universal'):
        total_processed+=1
        if count == N_INSTANCES_TO_GET:
            break
        for word in sentence:
            if word[1] == 'VERB' and LEMMATIZER.lemmatize(word[0].lower()) == 'build':
                left_filler = findleft(sentence.index(word),sentence)
                if left_filler is None:
                    #print("LEFT FILLER NONE")
                    continue
                right_filler = findright(sentence.index(word),sentence)
                if right_filler is None:
                    #print("RIGHT FILLER NONE")
                    continue
                couples.append([left_filler,right_filler])
                count +=1
    print("Found couples",count)
    print("Processed: ",total_processed)
    return couples

def pretty(d, indent=1):
   for key, value in d.items():
      print('\t' * indent + str(key)+": "+str(value))

def aggregate(fillers):
    """
    It groups fillers 2D array
    and print frequency of each filler
    """
    frequencies_left = {}
    frequencies_right = {}
    frequencies_couple = {}
    for filler in fillers:
        if filler[0]+"___"+filler[1] in frequencies_couple:
            frequencies_couple[filler[0]+"___"+filler[1]] += 1
        else:
            frequencies_couple[filler[0]+"___"+filler[1]] = 1

        if filler[0] in frequencies_left:
            frequencies_left[filler[0]] += 1
        else:
            frequencies_left[filler[1]] = 1

        if filler[1] in frequencies_right:
            frequencies_right[filler[1]] += 1
        else:
            frequencies_right[filler[1]] = 1
        
    print("LEFT: ")
    pretty(frequencies_left)
    print("RIGHT: ")
    pretty(frequencies_right)
    print("COUPLES: ")
    pretty(frequencies_couple)

def main():
    fillers = getFillers()
    #print(fillers)
    aggregate(fillers)


if __name__ == "__main__":
    #verbo scelto: COSTRUIRE
    #nltk.download('brown')
    #nltk.download('universal_tagset')
    main()