import spacy
from openie import StanfordOpenIE

# object and subject constants
OBJECT_DEPS = {"dobj", "dative", "attr", "oprd"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "agent", "expl"}

def main():

    '''

    Open Information Extraction example using Spacy
    - doc taken from https://en.wikipedia.org/wiki/World_War_II
    - comparative done with StanfordOpenIe library https://github.com/philipperemy/Stanford-OpenIE-Python

    '''

    
    text ="""The Empire of Japan aimed to dominate Asia and the Pacific and was 
        already at war with the Republic of China in 1937, but the world war is 
        generally said to have begun on 1 September 1939 with the invasion of 
        Poland by Germany and subsequent declarations of war on Germany by 
        France and the United Kingdom. From late 1939 to early 1941, in a 
        series of campaigns and treaties, Germany conquered or controlled much 
        of continental Europe, and formed the Axis alliance with Italy and 
        Japan. Under the Molotov-Ribbentrop Pact of August 1939, Germany and the 
        Soviet Union partitioned and annexed territories of their European 
        neighbours, Poland, Finland, Romania and the Baltic states. The war 
        continued primarily between the European Axis powers and the coalition 
        of the United Kingdom and the British Commonwealth, with campaigns 
        including the North Africa and East Africa campaigns, the aerial Battle 
        of Britain, the Blitz bombing campaign, the Balkan Campaign as well as 
        the long-running Battle of the Atlantic. In June 1941, the European Axis 
        powers launched an invasion of the Soviet Union, opening the largest 
        land theatre of war in history, which trapped the major part of the
        Axis' military forces into a war of attrition. In December 1941, Japan 
        attacked the United States and European territories in the Pacific
        Ocean, and quickly conquered much of the Western Pacific."""

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    sentences = list(doc.sents)
    for sentence in sentences:
        subject, verb, attribute = extract_svo(sentence,nlp)
        print("Subject: ",subject,"| Verb: ",verb,"| Obj: ",attribute)

    print("================ With StanfordOpenIe:")
    with StanfordOpenIE() as client:
        for sentence in sentences:    
            for triple in client.annotate(sentence.text):
                print('|-', triple)

# extract the subject, object and verb from the input
def extract_svo(doc,nlp):
    sub = []
    at = []
    ve = []
    for token in doc:
        # is this a stop word?
        if nlp.vocab[token.text].is_stop == True:
            continue
        # is this a verb?
        if token.pos_ == "VERB":
            ve.append(token.text)
        # is this the object?
        if token.dep_ in OBJECT_DEPS or token.head.dep_ in OBJECT_DEPS:
            at.append(token.text)
        # is this the subject?
        if token.dep_ in SUBJECT_DEPS or token.head.dep_ in SUBJECT_DEPS:
            sub.append(token.text)
    return " ".join(sub).strip().lower(), " ".join(ve).strip().lower(), " ".join(at).strip().lower()

if __name__ == "__main__":
    """
    In questa esercitazione ho implementato un semplice esempio di applicazione 
    della tecnica dell’Open Information Extraction su un testo preso da Wikipedia.
    """
    main()