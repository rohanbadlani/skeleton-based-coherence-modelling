import numpy as np
import scipy
from bert_embedding import BertEmbedding


class SentenceSimilarity(object):
    """
    Implements functionality to evaluate sentence similarity through various methods
    Can be used for sentence or skeleton similarity
    """
    def __init__(self, sentences):
        """
        Args:
        sentences (list of strings): sentences to init the similarity object with
        """
        
        self.sentences = sentences
        self.bert = BertEmbedding()
        self.bert_word_vecs = None
        self.bert_sent_vecs = None
        
    def get_bert_sent_vecs(self):
        self.bert_word_vecs = self.bert.embedding(self.sentences)
        # May have to optimize further
        self.bert_sent_vecs = []
        num_sentences = len(self.sentences)
        for idx in range(num_sentences):
            self.bert_sent_vecs.append([self.bert_word_vecs[idx][0], np.mean(
                self.bert_word_vecs[idx][1], axis=0)])
        self.bert_sent_vecs = np.array(self.bert_sent_vecs)

    def get_bert_cosine_similarity(self, idx1, idx2):
        sent1 = self.bert_sent_vecs[idx1][1]
        sent2 = self.bert_sent_vecs[idx2][1]
        cosine_distance = scipy.spatial.distance.cosine(sent1, sent2)
        cosine_similarity = 1.0 - cosine_distance
        return cosine_similarity

    def get_bert_euclidean_distance(self, idx1, idx2):
        sent1 = self.bert_sent_vecs[idx1][1]
        sent2 = self.bert_sent_vecs[idx2][1]
        euclidean_distance = scipy.spatial.distance.euclidean(sent1, sent2)
        return euclidean_distance

    
if __name__ == '__main__':
    bert_abstract = """We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.
 Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers.
 As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. 
BERT is conceptually simple and empirically powerful. 
It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7 (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5% absolute improvement), outperforming human performance by 2.0%."""
    sentences = bert_abstract.split("\n")
    similarity_object = SentenceSimilarity(sentences)
    similarity_object.get_bert_sent_vecs()
    print(similarity_object.get_bert_cosine_similarity(0, 1))
    print(similarity_object.get_bert_euclidean_distance(0, 1))
