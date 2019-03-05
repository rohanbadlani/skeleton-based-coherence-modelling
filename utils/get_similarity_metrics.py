import numpy as np
from sentence_similarity import SentenceSimilarity


def load_file(filename):
    """Load a file into a list of dicts."""
    paragraphs = []
    with open(filename, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            paragraph = json.loads(line)
            paragraphs.append(paragraph)
    return paragraphs


def separate_into_sents_and_skeletons(paragraphs):
    sentences = []
    skeletons = []
    for paragraph in paragraphs:
        sentences.extend(paragraphs['text'])
        skeletons.extend(paragraphs['skeletons'])
    return sentences, skeletons


def get_cosine_similarities(sentences):
    """Return a list of cosine similarities between pairs of consecutive sentences
    or skeletons
    """
    sim_object = SentenceSimilarity(sentences)
    sim_object.get_bert_sent_vecs()
    similarities = []
    for idx in range(len(sentences)-1):
        similarities.append(sim_object.get_bert_cosine_similarity(idx, idx+1))
    return similarities


if __name__ == '__main__':
    filename = 'train.txt'
    
