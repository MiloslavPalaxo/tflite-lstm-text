# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import tempfile
from collections import Counter

import numpy as np
import tensorflow as tf
from pathlib import Path

from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.tools import optimize_for_inference_lib

# Number of steps to train model.
TRAIN_STEPS = 100
TEST_SAMPLES = 5

CONFIG = tf.ConfigProto(device_count={"GPU": 0})

UNK_ID = 0
EOS_ID = 2
UNK = "<unk>"
SOS = "<s>"
EOS = '</s>'
DIR = "data/"


max_seq_len = 150
EMBED_SIZE = 64
DICT_SIZE = 10000


class BatchInput:
    """"""

    def __init__(self, data: np.array, batch_size: int, shuffle: bool):
        """Constructor for BatchInput"""
        self.data = data
        self.batch_cursor = 0
        self.batch_size = batch_size

        if shuffle:
            np.random.shuffle(data)

    def next_batch(self):
        self.batch_cursor += self.batch_size
        return self.data[self.batch_cursor - self.batch_size:self.batch_cursor]


class UnidirectionalSequenceLstmTest(test_util.TensorFlowTestCase):

    def setUp(self):
        tf.reset_default_graph()

        self.train_data = load_data_matrix(DIR + "data.csv")

        self.vocab_len = np.max(self.train_data)+1

        # Define constants
        # Unrolled through 28 time steps
        self.time_steps = 100
        # Learning rate for Adam optimizer
        self.learning_rate = 0.001
        # MNIST is meant to be classified in 10 classes(0-9).
        self.n_classes = 3
        # Batch size
        self.batch_size = 64
        # Lstm Units.
        self.num_units = 128

    def buildLstmLayer(self):
        return tf.keras.layers.StackedRNNCells([
            tf.lite.experimental.nn.TFLiteLSTMCell(
                self.num_units, use_peepholes=True, forget_bias=1.0, name="rnn1"),
            tf.lite.experimental.nn.TFLiteLSTMCell(
                self.num_units, num_proj=8, forget_bias=1.0, name="rnn2"),
            tf.lite.experimental.nn.TFLiteLSTMCell(
                self.num_units // 2,
                use_peepholes=True,
                num_proj=8,
                forget_bias=0,
                name="rnn3"),
            tf.lite.experimental.nn.TFLiteLSTMCell(
                self.num_units, forget_bias=1.0, name="rnn4")
        ])

    def buildModel(self, lstm_layer, is_dynamic_rnn):
        enc_embedding = tf.get_variable("emb", [self.vocab_len, EMBED_SIZE], tf.float32)

        # Weights and biases for output softmax layer.
        out_weights = tf.Variable(
            tf.random_normal([self.num_units, self.n_classes]))
        out_bias = tf.Variable(tf.random_normal([self.n_classes]))

        # input text placeholder
        sentences = tf.placeholder(
            tf.int32, [None, self.time_steps], name="INPUT_TEXT")

        lengths = tf.placeholder(tf.int32, [None], name="INPUT_LENGTH")

        encoder_emb_inp = tf.nn.embedding_lookup(enc_embedding, sentences)

        # x is shaped [batch_size,time_steps,num_inputs]
        if is_dynamic_rnn:
            lstm_input = tf.transpose(encoder_emb_inp, perm=[1, 0, 2])
            outputs, _ = tf.lite.experimental.nn.dynamic_rnn(
                lstm_layer, lstm_input, sequence_length=lengths, dtype="float32")
            outputs = tf.unstack(outputs, axis=0)
        else:
            lstm_input = tf.unstack(encoder_emb_inp, self.time_steps, 1)
            outputs, _ = tf.nn.static_rnn(lstm_layer, lstm_input, sequence_length=lengths,  dtype="float32")

        # Compute logits by multiplying outputs[-1] of shape [batch_size,num_units]
        # by the softmax layer's out_weight of shape [num_units,n_classes]
        # plus out_bias
        # TODO: change to last_encoder_state !!
        prediction = tf.matmul(outputs[-1], out_weights) + out_bias
        output_class = tf.nn.softmax(prediction, name="OUTPUT_CLASS")

        return sentences, lengths, prediction, output_class

    def trainModel(self, sentences, lengths, prediction, output_class, sess):
        batch_input = BatchInput(self.train_data, self.batch_size, True)

        # input label placeholder
        labels = tf.placeholder(tf.int32, [None])

        y = tf.one_hot(labels, 3)

        # Loss function
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        # Optimization
        opt = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(loss)

        # Initialize variables
        init = tf.global_variables_initializer()
        sess.run(init)
        for _ in range(TRAIN_STEPS):
            batch = batch_input.next_batch()

            batch_lengths = batch[:, 0]
            batch_sentences = batch[:, 1:self.time_steps + 1]
            batch_labels = batch[:, max_seq_len + 1]

            sess.run(opt, feed_dict={sentences: batch_sentences, lengths: batch_lengths, labels: batch_labels})

    def saveAndRestoreModel(self, lstm_layer, sess, saver, is_dynamic_rnn):
        model_dir = tempfile.mkdtemp()
        saver.save(sess, model_dir)

        # Reset the graph.
        tf.reset_default_graph()
        sentences, lengths, prediction, output_class = self.buildModel(
            lstm_layer, is_dynamic_rnn)

        new_sess = tf.Session(config=CONFIG)
        saver = tf.train.Saver()
        saver.restore(new_sess, model_dir)
        return sentences, lengths, prediction, output_class, new_sess

    def getInferenceResult(self, sentences, lengths, output_class, sess):
        # TODO: change to eval (may be)
        batch = BatchInput(self.train_data, self.batch_size, True).next_batch()

        batch_lengths = batch[0:TEST_SAMPLES, 0]
        batch_sentences = batch[0:TEST_SAMPLES, 1:self.time_steps + 1]

        expected_output = sess.run(output_class, feed_dict={sentences: batch_sentences, lengths: batch_lengths})
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, sess.graph_def, [output_class.op.name])
        return [batch_sentences, batch_lengths], expected_output, frozen_graph

    def tfliteInvoke(self, graph, test_inputs, outputs):
        tf.reset_default_graph()

        # input text placeholder
        sentences = tf.placeholder(
            tf.int32, [TEST_SAMPLES, self.time_steps], name="INPUT_TEXT_LITE")

        lengths = tf.placeholder(tf.int32, [TEST_SAMPLES], name="INPUT_LENGTH_LITE")

        tf.import_graph_def(graph, name="", input_map={"INPUT_TEXT": sentences, "INPUT_LENGTH": lengths})
        with tf.Session() as sess:
            curr = sess.graph_def
            curr = convert_op_hints_to_stubs(graph_def=curr)

        curr = optimize_for_inference_lib.optimize_for_inference(
            curr, ["INPUT_TEXT_LITE", "INPUT_LENGTH_LITE"], ["OUTPUT_CLASS"],
            [tf.int32.as_datatype_enum, tf.int32.as_datatype_enum])

        converter = tf.lite.TFLiteConverter(curr, [sentences, lengths], [outputs])
        tflite = converter.convert()
        interpreter = tf.lite.Interpreter(model_content=tflite)

        try:
            interpreter.allocate_tensors()
        except ValueError:
            assert False

        # first input (sentences)
        input_index = (interpreter.get_input_details()[0]["index"])
        interpreter.set_tensor(input_index, test_inputs[0])

        # second input (lengths)
        input_index = (interpreter.get_input_details()[1]["index"])
        interpreter.set_tensor(input_index, test_inputs[1])

        interpreter.invoke()
        output_index = (interpreter.get_output_details()[0]["index"])
        result = interpreter.get_tensor(output_index)
        # Reset all variables so it will not pollute other inferences.
        interpreter.reset_all_variables()
        return result

    def testStaticRnnMultiRnnCell(self):
        sess = tf.Session(config=CONFIG)

        sentences, lengths, prediction, output_class = self.buildModel(
            self.buildLstmLayer(), is_dynamic_rnn=False)
        self.trainModel(sentences, lengths, prediction, output_class, sess)

        saver = tf.train.Saver()
        sentences, lengths, prediction, output_class, new_sess = self.saveAndRestoreModel(
            self.buildLstmLayer(), sess, saver, is_dynamic_rnn=False)

        test_inputs, expected_output, frozen_graph = self.getInferenceResult(
            sentences, lengths, output_class, new_sess)

        result = self.tfliteInvoke(frozen_graph, test_inputs, output_class)
        self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))

    @test_util.enable_control_flow_v2
    def testDynamicRnnMultiRnnCell(self):
        sess = tf.Session(config=CONFIG)

        sentences, lengths, prediction, output_class = self.buildModel(
            self.buildLstmLayer(), is_dynamic_rnn=True)
        self.trainModel(sentences, lengths, prediction, output_class, sess)

        saver = tf.train.Saver()

        sentences, lengths, prediction, output_class, new_sess = self.saveAndRestoreModel(
            self.buildLstmLayer(), sess, saver, is_dynamic_rnn=True)

        test_inputs, expected_output, frozen_graph = self.getInferenceResult(
            sentences, lengths, output_class, new_sess)

        result = self.tfliteInvoke(frozen_graph, test_inputs, output_class)
        print(expected_output, result)
        self.assertTrue(np.allclose(expected_output, result, rtol=1e-6, atol=1e-2))


def load_data_matrix(file_name: str) -> np.array:
    return np.loadtxt(file_name, dtype=np.int32, delimiter=",")





if __name__ == "__main__":
    test.main()
