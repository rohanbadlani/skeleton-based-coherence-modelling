import numpy as np
import sys
import pdb
import ast
from random import shuffle

class SkeletonDataset(object):
	def __init__(self):
		self.stories = []

		self.sentence_repo = []

		self.story_sentences = []

		self.skeleton_list = []
		self.actual_text_list = []

	def check_matching_skeleton(self, skeleton_words, sentence_index, story_index):
		for word in skeleton_words:
			if word not in self.story_sentences[story_index][sentence_index]:
				return False
		return True

	def check_matching_skeleton(self, skeleton_words, sentence_tuple):
		sentence = sentence_tuple[0]
		for word in skeleton_words:
			#unicode strings are not getting compared properly. Handle later
			if word.isalnum() and word not in sentence:
				return False
		return True

	def checkSubstring(self, words, story):
		for word in words:
			#unicode strings are not getting compared properly. Handle later
			if word.isalnum() and word not in story:
				return False
		return True

	"""
	Test Code, might need later for cleaning dataset.

	def read_skeleton_dataset(self, story_filepath, sentence_filepath, skeleton_filepath):
		#read in all the stories
		with open(story_filepath) as fp:
			line = fp.readline()
			while line:
				self.stories.append(line)

				story = line.split("\n")[0]
				sentences = story.split(".")

				self.story_sentences.append([sent for sent in sentences if sent])
				#pdb.set_trace()

				line = fp.readline()

		with open(sentence_filepath) as fp:
			line = fp.readline()
			story_index = 0
			while line:
				sent_dict = ast.literal_eval(line)

				sentence = ' '.join(sent_dict["text"])

				if self.checkSubstring(sent_dict["text"], self.stories[story_index]):
					self.sentence_repo.append((sentence, story_index))
				else:
					story_index += 1
					#double check
					if self.checkSubstring(sent_dict["text"], self.stories[story_index]):
						self.sentence_repo.append((sentence, story_index))
					else:
						print ("Erroneus inputs")
						pdb.set_trace()

				line = fp.readline()

		pdb.set_trace()
		with open(skeleton_filepath) as fp:
			line = fp.readline()

			story_index = 0
			sentence_index = 0
			line_index = 0

			sent_skeletons = []

			while line:
				#process and identify which sentence it belongs to
				skeleton_words = line.split("\n")[0]
				skeleton_words = skeleton_words.split(" ")[:-1]
				
				#validation
				if not self.check_matching_skeleton(skeleton_words, self.sentence_repo[line_index]):
					print (line_index)
					pdb.set_trace()

				line_index += 1
				sent_skeletons.append(skeleton_words)

				sentence_index += 1
				if self.sentence_repo[line_index][1] == story_index:
					story_index += 1
					
					self.skeleton_list.append(sent_skeletons)
					sent_skeletons = []

				line = fp.readline()	
	"""
	def read_dataset(self, story_filepath, sentence_filepath, skeleton_filepath):
		#read in all the stories
		with open(story_filepath) as fp:
			line = fp.readline()
			while line:
				self.stories.append(line)

				story = line.split("\n")[0]
				sentences = story.split(".")

				self.story_sentences.append([sent for sent in sentences if sent])

				line = fp.readline()

		with open(sentence_filepath) as fp:
			line = fp.readline()
			story_index = 0
			while line:
				sent_dict = ast.literal_eval(line)

				sentence = ' '.join(sent_dict["text"])

				if self.checkSubstring(sent_dict["text"], self.stories[story_index]):
					self.sentence_repo.append((sentence, story_index))
				else:
					story_index += 1
					#double check
					if self.checkSubstring(sent_dict["text"], self.stories[story_index]):
						self.sentence_repo.append((sentence, story_index))
					else:
						print ("Erroneus inputs")
						pdb.set_trace()

				line = fp.readline()

		with open(skeleton_filepath) as fp:
			line = fp.readline()

			story_index = 0
			sentence_index = 0
			line_index = 0

			sent_skeletons = []
			sent_actual_text = []

			while line:
				#process and identify which sentence it belongs to
				skeleton_words = line.split("\n")[0]
				skeleton_words = skeleton_words.split(" ")[:-1]
				
				#validation
				if not self.check_matching_skeleton(skeleton_words, self.sentence_repo[line_index]):
					print (line_index)
					pdb.set_trace()

				sent_actual_text.append(self.sentence_repo[line_index][0])

				line_index += 1
				sent_skeletons.append(skeleton_words)
				
				sentence_index += 1

				if self.sentence_repo[line_index][1] != story_index:
					story_index += 1
					
					self.skeleton_list.append(sent_skeletons)
					self.actual_text_list.append(sent_actual_text)

					sent_skeletons = []
					sent_actual_text = []

				line = fp.readline()
		#pdb.set_trace()

	def dump_dataset_ordered(self, filepath):
		with open(filepath, "w") as fp:
			for skeletons, sentences in zip(self.skeleton_list, self.actual_text_list):
				json = {}
				json["text"] = sentences
				json["skeletons"] = skeletons
				fp.write(str(json) + "\n")
		fp.close()

	def dump_dataset_jumbled(self, filepath):
		with open(filepath, "w") as fp:
			for skeletons, sentences in zip(self.skeleton_list, self.actual_text_list):
				indices = [i for i in range(len(skeletons))]

				shuffle(indices)

				jumbled_sentences = []
				jumbled_skeletons = []

				for _, index in enumerate(indices):
					jumbled_sentences.append(sentences[index])
					jumbled_skeletons.append(skeletons[index])

				json = {}
				json["text"] = jumbled_sentences
				json["skeletons"] = jumbled_skeletons

				fp.write(str(json) + "\n")
		fp.close()

	def load_dataset(self, filepath):
		with open(filepath, "r") as fp:
			line = fp.readline()

			while line:
				jsondict = ast.literal_eval(line)
				
				skeletons = jsondict["skeletons"]
				sentences = jsondict["text"]

				skeleton_strings = [' '.join(skeleton) for skeleton in skeletons]
				self.skeleton_list.append(skeleton_strings)
				self.actual_text_list.append(sentences)

				line = fp.readline()
