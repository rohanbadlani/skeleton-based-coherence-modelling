import numpy as np
import sys
import pdb
import ast
from random import shuffle
import random
import string
import re

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
			if bool(re.match('^[a-zA-Z0-9]+$', word)) and word.isalnum() and word not in sentence:
				return False
		return True

	def checkSubstring(self, words, story):
		for word in words:
			#unicode strings are not getting compared properly. Handle later
			if bool(re.match('^[a-zA-Z0-9]+$', word)) and word.isalnum() and word not in story:
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

			while line and line_index < len(self.sentence_repo):
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

				if line_index < len(self.sentence_repo) and self.sentence_repo[line_index][1] != story_index:
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

	def construct_siamese_training_set(self, filepath):
		delimitter = ","
		with open(str(filepath) + "_skeletons", "w") as fp:
			fp.write(str(delimitter) + "sentences1" + str(delimitter) + "sentences2" + str(delimitter) + "is_similar\n")
			index = 0
			for i in range(len(self.skeleton_list)):
				story = self.skeleton_list[i]
				#loop over all possible combinations within the story
				for j in range(len(story)):
					for k in range(j+1, len(story)):
						sentence1 = story[j]
						sentence_1 = story[j]

						sentence2 = story[k]

						if len(sentence1) == 0 or len(sentence2) == 0:
							continue
						
						sentence1 = ' '.join(sentence1)
						sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
						
						sentence2 = ' '.join(sentence2)
						sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))

						fp.write(str(index) + str(delimitter) + str(sentence1) + str(delimitter) + str(sentence2) + str(delimitter) + str(1) + "\n")
						index += 1
						#find any other random story, not the current one and add a element
						random_story_index = random.choice(range(len(self.skeleton_list)))
						while random_story_index == i:
							random_story_index = random.choice(range(len(self.skeleton_list)))
						random_story = self.skeleton_list[random_story_index]

						sentence_opp = random_story[0]
						
						if len(sentence_1) == 0 or len(sentence_opp) == 0:
							continue
						
						sentence_opp = ' '.join(sentence_opp)
						sentence_opp = sentence_opp.translate(str.maketrans('', '', string.punctuation))

						fp.write(str(index) + str(delimitter) + str(sentence1) + str(delimitter) + str(sentence_opp) + str(delimitter) + str(0) + "\n")
						index += 1
		fp.close()

		with open(str(filepath) + "_sentences", "w") as fp:
			fp.write(str(delimitter) + "sentences1" + str(delimitter) + "sentences2" + str(delimitter) + "is_similar\n")
			index = 0
			for i in range(len(self.actual_text_list)):
				story = self.actual_text_list[i]
				#loop over all possible combinations within the story
				for j in range(len(story)):
					sentence1 = story[j]
					for k in range(j+1, len(story)):
						sentence2 = story[k]

						sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
						sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))

						if len(sentence1) == 0 or len(sentence2) == 0:
							continue

						fp.write(str(index) + str(delimitter) + str(sentence1) + str(delimitter) + str(sentence2) + str(delimitter) + str(1) + "\n")
						index += 1
						#find any other random story, not the current one and add a element
						random_story_index = random.choice(range(len(self.actual_text_list)))
						while random_story_index == i:
							random_story_index = random.choice(range(len(self.actual_text_list)))
						random_story = self.actual_text_list[random_story_index]

						sentence_opp = random_story[0]
						sentence_opp = sentence_opp.translate(str.maketrans('', '', string.punctuation))
						fp.write(str(index) + str(delimitter) + str(sentence1) + str(delimitter) + str(sentence_opp) + str(delimitter) + str(0) + "\n")
						index += 1
		fp.close()

                
	def construct_siamese_training_set_consecutive(self, filepath):
		delimitter = "\t"
		with open(str(filepath) + "_c_skeletons", "w") as fp:
			fp.write("sentences1" + str(delimitter) + "sentences2" + str(delimitter) + "is_similar\n")
			index = 0
			for i in range(len(self.skeleton_list)):
				story = self.skeleton_list[i]
				#loop over all possible combinations within the story
				for j in range(len(story)):
					sentence1 = story[j]
					if(j+1 < len(story) and len(sentence1) > 0):
						k = j+1

						sentence1 = story[j]
						sentence2 = story[k]

						if(len(sentence1) == 0 or len(sentence2) == 0):
							continue

						sentence1 = ' '.join(sentence1)
						sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
						
						sentence2 = ' '.join(sentence2)
						sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))

						sentence1 = sentence1.replace('\t', '')
						sentence1 = ' '.join(sentence1.split())
						if sentence1 == "":
							sentence1 = "empty"

						sentence2 = sentence2.replace('\t', '')
 						sentence2 = ' '.join(sentence2.split())
						if sentence2 == "":
							sentence2 = "empty"
						
						fp.write(str(sentence1) + str(delimitter) + str(sentence2) + str(delimitter) + str(1) + "\n")
						index += 1
						#find any other random story, not the current one and add a element
						random_story_index = random.choice(range(len(self.skeleton_list)))
						while random_story_index == i:
							random_story_index = random.choice(range(len(self.skeleton_list)))

						random_story = self.skeleton_list[random_story_index]

						sentence_opp = random_story[0]
						if sentence_opp == "":
							sentence_opp = "empty"

						if(len(sentence_opp) == 0):
							continue

						sentence_opp = ' '.join(sentence_opp)
						sentence_opp = sentence_opp.translate(str.maketrans('', '', string.punctuation))

						sentence_opp = sentence_opp.replace('\t', '')
						sentence_opp = ' '.join(sentence_opp.split())

						fp.write(str(sentence1) + str(delimitter) + str(sentence_opp) + str(delimitter) + str(0) + "\n")
						index += 1
		fp.close()

		with open(str(filepath) + "_c_sentences", "w") as fp:
			fp.write("sentences1" + str(delimitter) + "sentences2" + str(delimitter) + "is_similar\n")
			index = 0
			for i in range(len(self.actual_text_list)):
				story = self.actual_text_list[i]
				#loop over all possible combinations within the story
				for j in range(len(story)):
					sentence1 = story[j]
					if(j+1 < len(story)):
						k = j+1
						sentence2 = story[k]

						sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
						sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))

						if len(sentence1) == 0 or len(sentence2) == 0:
							continue

						sentence1 = sentence1.replace('\t', '')
						sentence1 = ' '.join(sentence1.split())
						if sentence1 == "":
							sentence1 = "empty"

						sentence2 = sentence2.replace('\t', '')
						sentence2 = ' '.join(sentence2.split())
						if sentence2 == "":
							sentence2 = "empty"
							
						fp.write(str(sentence1) + str(delimitter) + str(sentence2) + str(delimitter) + str(1) + "\n")
						index += 1
						#find any other random story, not the current one and add a element
						random_story_index = random.choice(range(len(self.actual_text_list)))
						while random_story_index == i:
							random_story_index = random.choice(range(len(self.actual_text_list)))
						random_story = self.actual_text_list[random_story_index]

						sentence_opp = random_story[0]
						sentence_opp = sentence_opp.translate(str.maketrans('', '', string.punctuation))
						if sentence_opp == "":
							sentence_opp = "empty"
							
						if len(sentence_opp) == 0:
							continue

						sentence_opp = sentence_opp.replace('\t', '')
						sentence_opp = ' '.join(sentence_opp.split())

						fp.write(str(sentence1) + str(delimitter) + str(sentence_opp) + str(delimitter) + str(0) + "\n")
						index += 1
		fp.close()

        def construct_siamese_test_set_paragraph(self, filepath):
                delimiter = "\t"
                with open(str(filepath) + "_c_skeletons", "w") as fp:
                        fp.write("sentences1" + str(delimitter) + "sentences2" + str(delimitter) + "is_similar\n")
			index = 0
                        for i in range(len(self.skeleton_list)):
                                story = self.skeleton_list[i]
                                rand_perm = np.random.permutation(len(story))
                                story_jumbled = []
                                for idx in rand_perm:
                                        story_jumbled.append(story[idx - 1])
                                for j in range(len(story)):
                                        sentence1 = story[j]
					if(j+1 < len(story) and len(sentence1) > 0):
						k = j+1

						sentence1 = story[j]
						sentence2 = story[k]

						if(len(sentence1) == 0 or len(sentence2) == 0):
							continue

						sentence1 = ' '.join(sentence1)
						sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
						
						sentence2 = ' '.join(sentence2)
						sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))

						sentence1 = sentence1.replace('\t', '')
						sentence1 = ' '.join(sentence1.split())
						if sentence1 == "":
							sentence1 = "empty"

						sentence2 = sentence2.replace('\t', '')
 						sentence2 = ' '.join(sentence2.split())
						if sentence2 == "":
							sentence2 = "empty"
						
						fp.write(str(sentence1) + str(delimitter) + str(sentence2) + str(delimitter) + str(1) + "\n")
						index += 1
                                for j in range(len(story_jumbled)):
                                        sentence1 = story_jumbled[j]
					if(j+1 < len(story_jumbled) and len(sentence1) > 0):
						k = j+1

						sentence1 = story_jumbled[j]
						sentence2 = story_jumbled[k]

						if(len(sentence1) == 0 or len(sentence2) == 0):
							continue

						sentence1 = ' '.join(sentence1)
						sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
						
						sentence2 = ' '.join(sentence2)
						sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))

						sentence1 = sentence1.replace('\t', '')
						sentence1 = ' '.join(sentence1.split())
						if sentence1 == "":
							sentence1 = "empty"

						sentence2 = sentence2.replace('\t', '')
 						sentence2 = ' '.join(sentence2.split())
						if sentence2 == "":
							sentence2 = "empty"
						
						fp.write(str(sentence1) + str(delimitter) + str(sentence2) + str(delimitter) + str(0) + "\n")
						index += 1
		fp.close()

                with open(str(filepath) + "_c_sentences", "w") as fp:
                        fp.write("sentences1" + str(delimitter) + "sentences2" + str(delimitter) + "is_similar\n")
			index = 0
                        for i in range(len(self.actual_text_list)):
                                story = self.actual_text_list[i]
                                rand_perm = np.random.permutation(len(story))
                                story_jumbled = []
                                for idx in rand_perm:
                                        story_jumbled.append(story[idx - 1])
                                for j in range(len(story)):
                                        sentence1 = story[j]
					if(j+1 < len(story) and len(sentence1) > 0):
						k = j+1

						sentence1 = story[j]
						sentence2 = story[k]

						if(len(sentence1) == 0 or len(sentence2) == 0):
							continue

						sentence1 = ' '.join(sentence1)
						sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
						
						sentence2 = ' '.join(sentence2)
						sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))

						sentence1 = sentence1.replace('\t', '')
						sentence1 = ' '.join(sentence1.split())
						if sentence1 == "":
							sentence1 = "empty"

						sentence2 = sentence2.replace('\t', '')
 						sentence2 = ' '.join(sentence2.split())
						if sentence2 == "":
							sentence2 = "empty"
						
						fp.write(str(sentence1) + str(delimitter) + str(sentence2) + str(delimitter) + str(1) + "\n")
						index += 1
                                for j in range(len(story_jumbled)):
                                        sentence1 = story_jumbled[j]
					if(j+1 < len(story_jumbled) and len(sentence1) > 0):
						k = j+1

						sentence1 = story_jumbled[j]
						sentence2 = story_jumbled[k]

						if(len(sentence1) == 0 or len(sentence2) == 0):
							continue

						sentence1 = ' '.join(sentence1)
						sentence1 = sentence1.translate(str.maketrans('', '', string.punctuation))
						
						sentence2 = ' '.join(sentence2)
						sentence2 = sentence2.translate(str.maketrans('', '', string.punctuation))

						sentence1 = sentence1.replace('\t', '')
						sentence1 = ' '.join(sentence1.split())
						if sentence1 == "":
							sentence1 = "empty"

						sentence2 = sentence2.replace('\t', '')
 						sentence2 = ' '.join(sentence2.split())
						if sentence2 == "":
							sentence2 = "empty"
						
						fp.write(str(sentence1) + str(delimitter) + str(sentence2) + str(delimitter) + str(0) + "\n")
						index += 1
		fp.close()
                
