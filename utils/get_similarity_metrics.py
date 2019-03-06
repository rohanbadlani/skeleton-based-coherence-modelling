import numpy as np
import scipy

from .sentence_similarity import SentenceSimilarity


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


def get_cosine_similarities(sentences, filename=''):
    """Return a list of cosine similarities between pairs of consecutive sentences
    or skeletons
    """
    sim_object = SentenceSimilarity(sentences)
    sim_object.get_bert_sent_vecs()
    if filename != '':
        np.save(filename, sim_object.bert_sent_vecs)
    similarities = []
    for idx in range(len(sentences)-1):
        similarities.append(sim_object.get_bert_cosine_similarity(idx, idx+1))
    return similarities


def get_cosine_sim_from_embs(embeddings):
    similarities = []
    nan_sims = 0
    # print(embeddings[0].shape)
    for idx in range(len(embeddings)-1):
        cosine_distance = scipy.spatial.distance.cosine(embeddings[idx], embeddings[idx+1])
        cosine_similarity = 1.0 - cosine_distance
        if np.sum(np.isfinite(embeddings[idx])) < len(embeddings[idx]):
            print(embeddings[idx])
            break
        # print(type(embeddings[idx]), type(embeddings[idx+1]))
        if np.sum(np.isfinite(embeddings[idx+1])) < len(embeddings[idx+1]):
            print(embeddings[idx+1])
            break
        
        # euclidean_distance = scipy.spatial.distance.euclidean(embeddings[idx], embeddings[idx+1])
        similarities.append(cosine_similarity)
        if similarities[-1] != similarities[-1]:
            # print(cosine_distance)
            nan_sims += 1
    print(nan_sims)
    return similarities
    

def get_aggregate_similarity(similarities):
    """
    Return an aggregate of similarities of sentences for a paragraph  level score
    Currently uses mean to aggregate. 
    """
    return np.mean(similarities)
    

if __name__ == '__main__':
    pass
    
