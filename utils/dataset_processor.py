from dataset import *


def main(story_filePath, sentence_filePath, skeleton_filePath, ordered_dataset_filepath):
	dataset = SkeletonDataset()
	dataset.read_dataset(story_filePath, sentence_filePath, skeleton_filePath)
	dataset.dump_dataset_ordered(ordered_dataset_filepath)
	dataset.dump_dataset_jumbled(jumbled_dataset_filepath)

	#Test loader
	#dataset.load_dataset(ordered_dataset_filepath)
	dataset.construct_siamese_training_set("siamese_data.txt")
	dataset.construct_siamese_training_set_consecutive("siamese_data.txt")

if(len(sys.argv)!=6):
    print("dataset_processor usage: Please provide the filepaths. Call this script as python dataset_processor.py <story-filapath> <senetence-filepath> <skeleton-filepath> <ordered_dataset_filepath> <jumbled_dataset_filepath>")
    sys.exit(1)

story_filePath = sys.argv[1]
sentence_filePath = sys.argv[2]
skeleton_filePath = sys.argv[3]
ordered_dataset_filepath = sys.argv[4]
jumbled_dataset_filepath = sys.argv[5]

main(story_filePath, sentence_filePath, skeleton_filePath, ordered_dataset_filepath)