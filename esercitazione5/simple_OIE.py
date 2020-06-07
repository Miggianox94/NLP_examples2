import spacy
from textpipeliner import PipelineEngine, Context
from textpipeliner.pipes import *

'''

Exercise based on library https://github.com/krzysiekfonal/textpipeliner

'''

def main():

    '''
    
    SequencePipe: Process the pipes in sequence, the result is propagated throught the chain
    AnyPipe: it simulates an OR. -> process the first pipe the return a non empty result 
    AggregatePipe: it simulates an AND. -> aggregate all the result of the inner pipes
    FindTokensPipe: This pipe takes a regex-like pattern to extract using the grammaregex library.
    NamedEntityFilterPipe: This pipe filters passed tokens choosing the ones which are part of a named entity.
    NamedEntityExtractorPipe: This pipe collects a whole chain from a single token which is part of an entity
    
    '''

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(
        "The Empire of Japan aimed to dominate Asia and the Pacific and was "
        "already at war with the Republic of China in 1937, but the world war is "
        "generally said to have begun on 1 September 1939 with the invasion of "
        "Poland by Germany and subsequent declarations of war on Germany by "
        "France and the United Kingdom. From late 1939 to early 1941, in a "
        "series of campaigns and treaties, Germany conquered or controlled much "
        "of continental Europe, and formed the Axis alliance with Italy and "
        "Japan. Under the Molotov-Ribbentrop Pact of August 1939, Germany and the "
        "Soviet Union partitioned and annexed territories of their European "
        "neighbours, Poland, Finland, Romania and the Baltic states. The war "
        "continued primarily between the European Axis powers and the coalition "
        "of the United Kingdom and the British Commonwealth, with campaigns "
        "including the North Africa and East Africa campaigns, the aerial Battle "
        "of Britain, the Blitz bombing campaign, the Balkan Campaign as well as "
        "the long-running Battle of the Atlantic. In June 1941, the European Axis "
        "powers launched an invasion of the Soviet Union, opening the largest "
        "land theatre of war in history, which trapped the major part of the "
        "Axis' military forces into a war of attrition. In December 1941, Japan "
        "attacked the United States and European territories in the Pacific "
        "Ocean, and quickly conquered much of the Western Pacific.")

    '''
    pipes_structure = [SequencePipe([FindTokensPipe("VERB/nsubj/NNP"),
                                 NamedEntityFilterPipe(),
                                 NamedEntityExtractorPipe()]),
                       AggregatePipe([FindTokensPipe("VERB"),
                                      FindTokensPipe("VERB/xcomp/VERB/aux/*"),
                                      FindTokensPipe("VERB/xcomp/VERB")]),
                       AnyPipe([FindTokensPipe("VERB/[acomp,amod]/ADJ"),
                                AggregatePipe([FindTokensPipe("VERB/[dobj,attr]/NOUN/det/DET"),
                                               FindTokensPipe("VERB/[dobj,attr]/NOUN/[acomp,amod]/ADJ")])])
                      ]
    '''
    pipes_structure = [
        SequencePipe([
            FindTokensPipe("VERB/nsubj/*"),
            NamedEntityFilterPipe(),
            NamedEntityExtractorPipe()
        ]),
        FindTokensPipe("VERB"),
        AnyPipe([
            SequencePipe([
                FindTokensPipe("VBD/dobj/NNP"),
                AggregatePipe([
                    NamedEntityFilterPipe("GPE"),
                    NamedEntityFilterPipe("PERSON")
                ]),
                NamedEntityExtractorPipe()
            ]),
            SequencePipe([
                FindTokensPipe("VBD/**/*/pobj/NNP"),
                AggregatePipe([
                    NamedEntityFilterPipe("LOC"),
                    NamedEntityFilterPipe("PERSON")
                ]),
                NamedEntityExtractorPipe()
            ])
        ])
    ]

    engine = PipelineEngine(pipes_structure, Context(doc), [0,1,2])
    print(engine.process())

if __name__ == "__main__":
    main()