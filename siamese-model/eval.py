#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper
import pdb
# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1552723808/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_string("eval_filepath", "validation.txt0", "Evaluate on this data (Default: None)")
tf.flags.DEFINE_string("vocab_filepath", "runs/1552723808/checkpoints/vocab", "Load training time vocabulary (Default: None)")
tf.flags.DEFINE_string("model", "runs/1552723808/checkpoints/model-995", "Load trained model checkpoint (Default: None)")
tf.flags.DEFINE_string("accuracy_results_filepath", "results/output_accuracy.txt", "Load trained model checkpoint (Default: None)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#Utilities
def get_accuracy_evaluation_metric_sentence_level(all_predictions):
    """
    Looks at 2 consecutive predictions and assumes 1st one is from ordered and 2nd is from unordered. 
    Counts 1 if 1st > 2nd, else 0, and finally reports counts / total_pairs_of_sentences 
    """
    pair_count = 0
    success_count = 0
    index = 0
    while (index + 1) < len(all_predictions):
        ordered_sentence_pair_pred = all_predictions[index]
        unordered_sentence_pair_pred = all_predictions[index + 1]
        if ordered_sentence_pair_pred >= unordered_sentence_pair_pred:
            success_count += 1
        pair_count += 1
        index += 2

    if (pair_count != 0):
        return success_count / float(pair_count)
    else:
        return 0.0

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if FLAGS.eval_filepath==None or FLAGS.vocab_filepath==None or FLAGS.model==None :
    print("Eval or Vocab filepaths are empty.")
    exit()

# load data and map id-transform based on training time vocabulary
inpH = InputHelper()
x1_test,x2_test,y_test = inpH.getTestDataSet(FLAGS.eval_filepath, FLAGS.vocab_filepath, 30)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = FLAGS.model
print(checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
        input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]

        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/distance").outputs[0]

        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]

        sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

        #emb = graph.get_operation_by_name("embedding/W").outputs[0]
        #embedded_chars = tf.nn.embedding_lookup(emb,input_x)
        # Generate batches for one epoch
        batches = inpH.batch_iter(list(zip(x1_test,x2_test,y_test)), 2*FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_d=[]
        for db in batches:
            try:
                x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
                batch_predictions, batch_acc, batch_sim = sess.run([predictions,accuracy,sim], {input_x1: x1_dev_b, input_x2: x2_dev_b, input_y:y_dev_b, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                print(batch_predictions)
                all_d = np.concatenate([all_d, batch_sim])
                print("DEV acc {}".format(batch_acc))
            except:
                print ("Input had some issues. Ignoring that batch and continuing.")
        for ex in all_predictions:
            print (ex)
        correct_predictions = float(np.mean(all_d == y_test))
        print("Prediction Accuracy: {:g}".format(correct_predictions))

        evaluation_metric_result = get_accuracy_evaluation_metric_sentence_level(all_predictions)
        print("Evaluation Metric (Sentence Level) Accuracy: {:g}".format(evaluation_metric_result))

        output_filepath = FLAGS.accuracy_results_filepath
        with open(output_filepath, "w") as fp:
            fp.write("Prediction Accuracy: {:g}".format(correct_predictions) + "\n")    
            fp.write("Evaluation Metric (Sentence Level) Accuracy: {:g}".format(evaluation_metric_result) + "\n")
            
        fp.close()
