import numpy as np

from utils.dataset import SkeletonDataset
from utils.get_similarity_metrics import *
from utils.sentence_similarity import *


def load_data(filename):
    dataset = SkeletonDataset()
    with open(filename, 'r') as fp:
        dataset.load_dataset(filename)
    return dataset
    

def get_agg_scores_from_embs(para_lengths):
    """
    Get aggregate similarity scores and other metrics from embeddings files. 
    """
    ordered_embs = np.array(np.load('ordered_1000.npy')[:, 1])
    # ordered_embs = np.reshape(ordered_embs, [len(ordered_embs), len(ordered_embs[0])])
    ordered_sk_embs = np.array(np.load('ordered_sk_1000.npy')[:, 1])
    # ordered_sk_embs = np.reshape(ordered_sk_embs, [len(ordered_embs), len(ordered_embs[0])])
    jumbled_embs = np.array(np.load('jumbled_1000.npy')[:, 1])
    jumbled_sk_embs = np.array(np.load('jumbled_sk_1000.npy')[:, 1])

    print(ordered_sk_embs.shape)
    ordered_sk_embs_valid = []
    for i in range(len(ordered_sk_embs)):
        if isinstance(ordered_sk_embs[i], np.ndarray):
            ordered_sk_embs_valid.append(ordered_sk_embs[i])
    ordered_sk_embs = np.array(ordered_sk_embs_valid)
    print(ordered_sk_embs.shape)
    rand_perm = np.random.permutation(len(ordered_sk_embs))
    jumbled_sk_embs = ordered_sk_embs[rand_perm]
    rand_perm = np.random.permutation(len(ordered_embs))
    jumbled_embs = jumbled_embs[rand_perm]
    print(jumbled_sk_embs.shape)
    
    ordered_similarities = np.array(get_cosine_sim_from_embs(ordered_embs))
    print("1")
    ordered_sk_similarities = np.array(get_cosine_sim_from_embs(ordered_sk_embs))
    print("2")
    jumbled_similarities = np.array(get_cosine_sim_from_embs(jumbled_embs))
    print("3")
    jumbled_sk_similarities = np.array(get_cosine_sim_from_embs(jumbled_sk_embs))

    print(ordered_similarities.shape, jumbled_similarities.shape, ordered_sk_similarities.shape, jumbled_sk_similarities.shape)
    sents_correct_preds, skeletons_correct_preds = 0, 0
    
    cur_length = 0
    ordered_tot, ordered_sk_tot, jumbled_tot, jumbled_sk_tot = 0.0, 0.0, 0.0, 0.0
    
    # for length in para_lengths:
    #     ordered_agg = get_aggregate_similarity(ordered_similarities[cur_length:cur_length+length-1])
    #     jumbled_agg = get_aggregate_similarity(jumbled_similarities[cur_length:cur_length+length-1])
    #     ordered_sk_agg = get_aggregate_similarity(ordered_sk_similarities[cur_length:cur_length+length-1])
    #     jumbled_sk_agg = get_aggregate_similarity(jumbled_sk_similarities[cur_length:cur_length+length-1])
    #     cur_length += length

    #     if length != 1:
    #         ordered_tot += ordered_agg
    #         jumbled_tot += jumbled_agg
            
    #         if ordered_agg >= jumbled_agg:
    #             sents_correct_preds += 1
    #             # print("right  ",  idx)
    #         if ordered_sk_agg == ordered_sk_agg:
    #             ordered_sk_tot += ordered_sk_agg
    #             jumbled_sk_tot += jumbled_sk_agg
    #             if ordered_sk_agg >= jumbled_sk_agg:
    #                 skeletons_correct_preds += 1
    #                 # print("right:", idx)
    print(jumbled_similarities[:100])
    correct_ordered = np.sum(ordered_similarities >= 0.5)
    correct_jumbled = np.sum(jumbled_similarities < 0.5)
    correct_ordered_sk = np.sum(ordered_sk_similarities >= 0.5)
    correct_jumbled_sk = np.sum(jumbled_sk_similarities < 0.5)
    
    correct = ordered_similarities >= jumbled_similarities
    print(correct.shape)
    correct_sk = ordered_sk_similarities >= jumbled_sk_similarities
    print("correct by style 1: ", correct_ordered, correct_jumbled, correct_ordered_sk, correct_jumbled_sk)
    print(correct_sk.shape)
    print(np.sum(correct), np.sum(correct_sk))
    print(np.sum(ordered_similarities), np.sum(jumbled_similarities))
    print(np.sum(ordered_sk_similarities), np.sum(jumbled_sk_similarities))
    
    # print(sents_correct_preds)
    # print(skeletons_correct_preds)
    # print(ordered_tot, jumbled_tot)
    # print(ordered_sk_tot, jumbled_sk_tot)


def main():
    ordered_text_filename = '../ordered_set.txt'
    jumbled_text_filename = '../jumbled_set.txt'
    
    ordered_data = load_data(ordered_text_filename)
    jumbled_data = load_data(jumbled_text_filename)
    print(len(ordered_data.actual_text_list))
    print(len(ordered_data.skeleton_list))
    # print(ordered_data.skeleton_list[0][1])
    # print(len(ordered_data.actual_text_list[0]))
    # print(ordered_data.actual_text_list[0][0])

    sents_correct_preds = 0
    skeletons_correct_preds = 0

    ordered_sent_list = []
    ordered_skeleton_list = []
    jumbled_sent_list = []
    jumbled_skeleton_list = []
    para_lengths = []
    
    for idx in range(1000):
        ordered_sent_list.extend(ordered_data.actual_text_list[idx])
        ordered_skeleton_list.extend(ordered_data.skeleton_list[idx])
        jumbled_sent_list.extend(jumbled_data.actual_text_list[idx])
        jumbled_skeleton_list.extend(jumbled_data.skeleton_list[idx])
        para_lengths.append(len(ordered_data.actual_text_list[idx]))
        
    get_agg_scores_from_embs(para_lengths)
    
    # ordered_similarities = get_cosine_similarities(ordered_sent_list, 'ordered_1000')
    # jumbled_similarities = get_cosine_similarities(jumbled_sent_list, 'jumbled_1000')
    # ordered_sk_similarities = get_cosine_similarities(ordered_skeleton_list, 'ordered_sk_1000')
    # jumbled_sk_similarities = get_cosine_similarities(jumbled_skeleton_list, 'jumbled_sk_1000')

    # cur_length = 0
    # for length in para_lengths:
    #     ordered_agg = get_aggregate_similarity(ordered_similarities[cur_length:cur_length+length-1])
    #     jumbled_agg = get_aggregate_similarity(jumbled_similarities[cur_length:cur_length+length-1])
    #     ordered_sk_agg = get_aggregate_similarity(ordered_sk_similarities[cur_length:cur_length+length-1])
    #     jumbled_sk_agg = get_aggregate_similarity(jumbled_sk_similarities[cur_length:cur_length+length-1])
    #     cur_length += length
    #     if ordered_agg >= jumbled_agg:
    #         sents_correct_preds += 1
    #         print("right  ",  idx)
    #     if ordered_sk_agg >= jumbled_sk_agg:
    #         skeletons_correct_preds += 1
    #         print("right:", idx)
            
    # print(sents_correct_preds)
    # print(skeletons_correct_preds)

    
if __name__ == '__main__':
    main()
