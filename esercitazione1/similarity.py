import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 


STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def computeSimilarity(first, second):
    """ritorna la similarità tra due liste di token"""
    return len(set(first).intersection(set(second)))/min(len(first),len(second))

def main():
    """
    Questa esercitazione prevede il calcolo di similarità tra le definizioni date per 4 concetti mostrati a lezione
    """
    
    #read csv
    df = pd.read_csv('definizioni.csv')
    #print(df)

    #############lemming and stop words elimination
    buildingList = df['concreto_generico_building'].tolist()
    buildinglistFiltered = []
    for sentence in buildingList:
        token = word_tokenize(sentence)
        filtered_sentence = [LEMMATIZER.lemmatize(w.lower()) for w in token if (isinstance(w,str)) and (not w in STOP_WORDS) and w.isalpha()]
        buildinglistFiltered.append(filtered_sentence)

    freedomList = df['astratto_generico_freedom'].tolist()
    freedomList = [x for x in freedomList if str(x) != 'nan']
    freedomlistFiltered = []
    for sentence in freedomList:
        token = word_tokenize(sentence)
        filtered_sentence = [LEMMATIZER.lemmatize(w.lower()) for w in token if (isinstance(w,str)) and (not w in STOP_WORDS) and w.isalpha()]
        freedomlistFiltered.append(filtered_sentence)

    moleculeList = df['concreto_specifico_molecule'].tolist()
    moleculelistFiltered = []
    for sentence in moleculeList:
        token = word_tokenize(sentence)
        filtered_sentence = [LEMMATIZER.lemmatize(w.lower()) for w in token if (isinstance(w,str)) and (not w in STOP_WORDS) and w.isalpha()]
        moleculelistFiltered.append(filtered_sentence)

    compassionList = df['astratto_specifico_compassion'].tolist()
    compassionList = [x for x in compassionList if str(x) != 'nan']
    compassionlistFiltered = []
    for sentence in compassionList:
        token = word_tokenize(sentence)
        filtered_sentence = [LEMMATIZER.lemmatize(w.lower()) for w in token if (isinstance(w,str)) and (not w in STOP_WORDS) and w.isalpha()]
        compassionlistFiltered.append(filtered_sentence)

    #############calculate avg similarity
    i = 0
    sim_building = 0
    counter = 0
    while i < len(buildinglistFiltered):
        j = i+1
        while j < len(buildinglistFiltered):
            sim_building += computeSimilarity(buildinglistFiltered[i],buildinglistFiltered[j])
            j+=1
            counter+=1
        i+=1
    sim_building_avg = sim_building/counter
    print("AVG SIMILARITY BUILDING: ",sim_building_avg)

    i = 0
    sim_freedom = 0
    counter = 0
    while i < len(freedomlistFiltered):
        j = i+1
        while j < len(freedomlistFiltered):
            sim_freedom += computeSimilarity(freedomlistFiltered[i],freedomlistFiltered[j])
            j+=1
            counter+=1
        i+=1
    sim_freedom_avg = sim_freedom/counter
    print("AVG SIMILARITY FREEDOM: ",sim_freedom_avg)

    i = 0
    sim_molecule = 0
    counter = 0
    while i < len(moleculelistFiltered):
        j = i+1
        while j < len(moleculelistFiltered):
            sim_molecule += computeSimilarity(moleculelistFiltered[i],moleculelistFiltered[j])
            j+=1
            counter+=1
        i+=1
    sim_molecule_avg = sim_molecule/counter
    print("AVG SIMILARITY MOLECULE: ",sim_molecule_avg)

    i = 0
    sim_compassion = 0
    counter = 0
    while i < len(compassionlistFiltered):
        j = i+1
        while j < len(compassionlistFiltered):
            sim_compassion += computeSimilarity(compassionlistFiltered[i],compassionlistFiltered[j])
            j+=1
            counter+=1
        i+=1
    sim_compassion_avg = sim_compassion/counter
    print("AVG SIMILARITY COMPASSION: ",sim_compassion_avg)

    


if __name__ == "__main__":
    main()