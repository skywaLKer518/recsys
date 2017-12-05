from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope

import data_iterator
# import env


class SeqModel(object):

    def __init__(self,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 embeddingAttribute,
                 withAdagrad=True,
                 num_samples=512,
                 forward_only=False,
                 dropoutRate=1.0,
                 START_ID=0,
                 loss="ce",
                 devices="",
                 run_options=None,
                 run_metadata=None,
                 use_concat=True,
                 output_feat=1,
                 no_input_item_feature=False,
                 no_user_id=True,
                 topk_n=30,
                 dtype=tf.float32):
        """Create the model.

        Args:
        buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
        size: number of units in each layer of the model.
        num_layers: number of layers in the model.
        max_gradient_norm: gradients will be clipped to maximally this norm.
        batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.

        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rate by this much when needed.

        num_samples: number of samples for sampled softmax.
        forward_only: if set, we do not construct the backward pass in the model.
        dtype: the data type to use to store internal variables.
        """
        self.embeddingAttribute = embeddingAttribute
        self.buckets = buckets
        self.START_ID = START_ID
        self.PAD_ID = START_ID
        self.USER_PAD_ID = 0
        self.batch_size = batch_size
        self.loss = loss
        self.devices = devices
        self.run_options = run_options
        self.run_metadata = run_metadata
        self.output_feat = output_feat
        self.no_input_item_feature = no_input_item_feature
        self.topk_n = topk_n
        self.dtype = dtype

        with tf.device(devices[0]):
            self.dropoutRate = tf.Variable(
                float(dropoutRate), trainable=False, dtype=dtype)
            self.dropoutAssign_op = self.dropoutRate.assign(dropoutRate)
            self.dropout10_op = self.dropoutRate.assign(1.0)
            self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=dtype)
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)

        with tf.device(devices[1]):
            single_cell = tf.contrib.rnn.LSTMCell(size, state_is_tuple=True)
            single_cell = tf.contrib.rnn.DropoutWrapper(
                single_cell, input_keep_prob=self.dropoutRate)
            if num_layers >= 1:
                single_cell = tf.contrib.rnn.MultiRNNCell(
                    [single_cell] * num_layers, state_is_tuple=True)
            single_cell = tf.contrib.rnn.DropoutWrapper(
                single_cell, output_keep_prob=self.dropoutRate)

        self.single_cell = single_cell

        # Feeds for inputs.
        with tf.device(devices[2]):
            self.targets = []
            self.target_ids = []
            self.target_weights = []

            # target: 1  2  3  4
            # inputs: go 1  2  3
            # weights:1  1  1  1

            for i in xrange(buckets[-1]):
                self.targets.append(tf.placeholder(tf.int32,
                                                   shape=[self.batch_size], name="target{}".format(i)))
                self.target_ids.append(tf.placeholder(tf.int32,
                                                      shape=[self.batch_size], name="target_id{}".format(i)))
                self.target_weights.append(tf.placeholder(dtype,
                                                          shape=[self.batch_size], name="target_weight{}".format(i)))

        with tf.device(devices[0]):

            self.inputs = []

            if use_concat:
                user_embed, _ = self.embeddingAttribute.get_batch_user(
                    1.0, concat=True, no_id=no_user_id)
                user_embed_size = self.embeddingAttribute.get_user_model_size(
                    no_id=no_user_id, concat=True)
                item_embed_size = self.embeddingAttribute.get_item_model_size(
                    concat=True)
                w_input_user = tf.get_variable(
                    "w_input_user", [user_embed_size, size], dtype=dtype)
                w_input_item = tf.get_variable(
                    "w_input_item", [item_embed_size, size], dtype=dtype)
                user_embed_transform = tf.matmul(user_embed, w_input_user)

                for i in xrange(buckets[-1]):
                    name = "input{}".format(i)
                    item_embed, _ = self.embeddingAttribute.get_batch_item(name,
                                                                           self.batch_size, concat=True, no_attribute=self.no_input_item_feature)
                    item_embed_transform = tf.matmul(item_embed, w_input_item)
                    input_embed = user_embed_transform + item_embed_transform
                    self.inputs.append(input_embed)
            else:
                user_embed, _ = self.embeddingAttribute.get_batch_user(
                    1.0, concat=False, no_id=no_user_id)

                for i in xrange(buckets[-1]):
                    name = "input{}".format(i)
                    item_embed, _ = self.embeddingAttribute.get_batch_item(name,
                                                                           self.batch_size, concat=False, no_attribute=self.no_input_item_feature)
                    item_embed = tf.reduce_mean(item_embed, 0)
                    input_embed = tf.reduce_mean([user_embed, item_embed], 0)
                    self.inputs.append(input_embed)

        self.outputs, self.losses, self.outputs_full, self.losses_full, self.topk_values, self.topk_indexes = self.model_with_buckets(
            self.inputs, self.targets, self.target_weights, self.buckets, single_cell, self.embeddingAttribute, dtype, devices=devices)

        # for warp
        if self.loss in ["warp", "mw"]:
            self.set_mask, self.reset_mask = self.embeddingAttribute.get_warp_mask(
                device=self.devices[2])

        # with tf.device(devices[0]):
        # train
        with tf.device(devices[0]):
            params = tf.trainable_variables()
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                self.gradient_norms = []
                self.updates = []
                if withAdagrad:
                    opt = tf.train.AdagradOptimizer(self.learning_rate)
                else:
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)

                for b in xrange(len(buckets)):
                    gradients = tf.gradients(
                        self.losses[b], params, colocate_gradients_with_ops=True)
                    clipped_gradients, norm = tf.clip_by_global_norm(
                        gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.global_variables())

    def init_beam_decoder(self, beam_size=10, max_steps=30):

        # a non bucket design
        #
        # how to feed in:
        # user_history = [1,2,3,4]
        # inputs = [GO, 1, 2, 3], sequene_length = [4-1]

        self.beam_size = beam_size

        init_state = self.single_cell.zero_state(1, self.dtype)
        self.before_state = []
        self.after_state = []
        print(init_state)
        shape = [self.beam_size, init_state[0].c.get_shape()[1]]

        with tf.device(self.devices[0]):

            with tf.variable_scope("beam_search"):

                # two variable: before_state, after_state
                for i, state_tuple in enumerate(init_state):
                    cb = tf.get_variable("before_c_{}".format(
                        i), shape, initializer=tf.constant_initializer(0.0), trainable=False)
                    hb = tf.get_variable("before_h_{}".format(
                        i), shape, initializer=tf.constant_initializer(0.0), trainable=False)
                    sb = tf.contrib.rnn.LSTMStateTuple(cb, hb)
                    ca = tf.get_variable("after_c_{}".format(
                        i), shape, initializer=tf.constant_initializer(0.0), trainable=False)
                    ha = tf.get_variable("after_h_{}".format(
                        i), shape, initializer=tf.constant_initializer(0.0), trainable=False)
                    sa = tf.contrib.rnn.core_rnn_cell.LSTMStateTuple(ca, ha)
                    self.before_state.append(sb)
                    self.after_state.append(sa)

                # a new place holder for sequence_length
                self.sequence_length = tf.placeholder(
                    tf.int32, shape=[1], name="sequence_length")

                # the final_state after processing the start state
            with tf.variable_scope("", reuse=True):
                _, self.beam_final_state = tf.contrib.rnn.static_rnn(
                    self.single_cell, self.inputs, initial_state=init_state, sequence_length=self.sequence_length)

            with tf.variable_scope("beam_search"):
                # copy the final_state to before_state
                self.final2before_ops = []  # an operation sequence
                for i in xrange(len(self.before_state)):
                    final_c = self.beam_final_state[i].c
                    final_h = self.beam_final_state[i].h
                    final_c_expand = tf.nn.embedding_lookup(
                        final_c, [0] * self.beam_size)
                    final_h_expand = tf.nn.embedding_lookup(
                        final_h, [0] * self.beam_size)
                    copy_c = self.before_state[i].c.assign(final_c_expand)
                    copy_h = self.before_state[i].h.assign(final_h_expand)
                    self.final2before_ops.append(copy_c)
                    self.final2before_ops.append(copy_h)

                # operation: copy after_state to before_state according to a ma
                self.beam_parent = tf.placeholder(
                    tf.int32, shape=[self.beam_size], name="beam_parent")
                self.after2before_ops = []  # an operation sequence
                for i in xrange(len(self.before_state)):
                    after_c = self.after_state[i].c
                    after_h = self.after_state[i].h
                    after_c_expand = tf.nn.embedding_lookup(
                        after_c, self.beam_parent)
                    after_h_expand = tf.nn.embedding_lookup(
                        after_h, self.beam_parent)
                    copy_c = self.before_state[i].c.assign(after_c_expand)
                    copy_h = self.before_state[i].h.assign(after_h_expand)
                    self.after2before_ops.append(copy_c)
                    self.after2before_ops.append(copy_h)

            # operation: one step RNN
            with tf.variable_scope("", reuse=True):
                self.beam_step_outputs, self.beam_step_state = tf.contrib.rnn.static_rnn(
                    self.single_cell, self.beam_step_inputs, initial_state=self.before_state)

            with tf.variable_scope("beam_search"):
                # operate: copy beam_step_state to after_state
                self.beam2after_ops = []  # an operation sequence
                for i in xrange(len(self.after_state)):
                    copy_c = self.after_state[i].c.assign(
                        self.beam_step_state[i].c)
                    copy_h = self.after_state[i].h.assign(
                        self.beam_step_state[i].h)
                    self.beam2after_ops.append(copy_c)
                    self.beam2after_ops.append(copy_h)

    def show_before_state(self):
        for i in xrange(self.before_state):
            print(self.before_state[i].c.eval())
            print(self.before_state[i].h.eval())

    def beam_step(self, session, index=0, beam_input=None, user_input=None, item_inputs=None, sequence_length=None, bucket_id=0):
        if index == 0:
            length = self.buckets[bucket_id]

            input_feed = {}
            (update_sampled, input_feed_sampled, input_feed_warp) = self.embeddingAttribute.add_input(
                input_feed, user_input, item_inputs, forward_only=True, recommend=True)
            input_feed[self.sequence_length.name] = sequence_length

            output_feed = [self.final2before.ops]

            self.show_before_state()
            _ = session.run(output_feed, input_feed)
            self.show_before_state()

        else:
            pass

    def step(self, session, user_input, item_inputs, targets, target_weights,
             bucket_id, item_sampled=None, item_sampled_id2idx=None, forward_only=False, recommend=False):

        length = self.buckets[bucket_id]

        targets_mapped = self.embeddingAttribute.target_mapping(targets)
        input_feed = {}
        for l in xrange(length):
            input_feed[self.targets[l].name] = targets_mapped[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            if self.loss in ['mw', 'ce']:
                input_feed[self.target_ids[l].name] = targets[l]

        (update_sampled, input_feed_sampled, input_feed_warp) = self.embeddingAttribute.add_input(input_feed, user_input,
                                                                                                  item_inputs, forward_only=forward_only, recommend=recommend, loss=self.loss, item_sampled_id2idx=item_sampled_id2idx)
        if self.loss in ["warp", "mw"]:
            session.run(self.set_mask[self.loss], input_feed_warp)

        if item_sampled is not None and self.loss in ['mw', 'mce']:
            session.run(update_sampled, input_feed_sampled)

        # output_feed
        if forward_only:
            output_feed = [self.losses_full[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            output_feed += [self.updates[bucket_id],
                            self.gradient_norms[bucket_id]]

        if self.loss in ["warp", "mw"]:
            session.run(self.set_mask[self.loss], input_feed_warp)

        outputs = session.run(
            output_feed, input_feed, options=self.run_options, run_metadata=self.run_metadata)

        if self.loss in ["warp", "mw"]:
            session.run(self.reset_mask[self.loss], input_feed_warp)

        return outputs[0]

    def step_recommend(self, session, user_input, item_inputs, positions, bucket_id):
        length = self.buckets[bucket_id]
        if bucket_id == 0:
            pre_length = 0
        else:
            pre_length = self.buckets[bucket_id - 1]

        input_feed = {}

        (update_sampled, input_feed_sampled, input_feed_warp) = self.embeddingAttribute.add_input(
            input_feed, user_input, item_inputs, forward_only=True, recommend=True, loss=self.loss)

        # output_feed
        output_feed = {}

        for pos in range(pre_length, length):
            output_feed[pos] = [self.topk_values[bucket_id]
                                [pos], self.topk_indexes[bucket_id][pos]]

        outputs = session.run(
            output_feed, input_feed, options=self.run_options, run_metadata=self.run_metadata)

        # results = [(uid, [value], [index])]
        results = []
        for i, pos in enumerate(positions):
            uid = user_input[i]
            values = outputs[pos][0][i, :]
            indexes = outputs[pos][1][i, :]
            results.append((uid, values, indexes))

        return results

    def get_batch(self, data_set, bucket_id, start_id=None):
        length = self.buckets[bucket_id]

        users, item_inputs, item_outputs, weights = [], [], [], []

        for i in xrange(self.batch_size):
            if start_id == None:
                user, item_seq = random.choice(data_set[bucket_id])
            else:
                if start_id + i < len(data_set[bucket_id]):
                    user, item_seq = data_set[bucket_id][start_id + i]
                else:
                    user = self.USER_PAD_ID
                    item_seq = []

            pad_seq = [self.PAD_ID] * (length - len(item_seq))
            if len(item_seq) == 0:
                item_input_seq = [self.START_ID] + pad_seq[1:]
            else:
                item_input_seq = [self.START_ID] + item_seq[:-1] + pad_seq
            item_output_seq = item_seq + pad_seq
            target_weight = [1.0] * len(item_seq) + [0.0] * len(pad_seq)

            users.append(user)
            item_inputs.append(item_input_seq)
            item_outputs.append(item_output_seq)
            weights.append(target_weight)

        # Now we create batch-major vectors from the data selected above.
        def batch_major(l):
            output = []
            for i in xrange(len(l[0])):
                temp = []
                for j in xrange(self.batch_size):
                    temp.append(l[j][i])
                output.append(temp)
            return output

        batch_user = users
        batch_item_inputs = batch_major(item_inputs)
        batch_item_outputs = batch_major(item_outputs)
        batch_weights = batch_major(weights)

        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set[bucket_id]):
            finished = True

        return batch_user, batch_item_inputs, batch_item_outputs, batch_weights, finished

    def get_batch_recommend(self, data_set, bucket_id, start_id=None):
        length = self.buckets[bucket_id]

        users, item_inputs, positions, valids = [], [], [], []

        for i in xrange(self.batch_size):
            if start_id == None:
                user, item_seq = random.choice(data_set[bucket_id])
                valid = 1
                position = len(item_seq) - 1
            else:
                if start_id + i < len(data_set[bucket_id]):
                    user, item_seq = data_set[bucket_id][start_id + i]
                    valid = 1
                    position = len(item_seq) - 1
                else:
                    user = self.USER_PAD_ID
                    item_seq = []
                    valid = 0
                    position = length - 1

            pad_seq = [self.PAD_ID] * (length - len(item_seq))
            item_input_seq = item_seq + pad_seq
            valids.append(valid)
            users.append(user)
            positions.append(position)
            item_inputs.append(item_input_seq)

        # Now we create batch-major vectors from the data selected above.
        def batch_major(l):
            output = []
            for i in xrange(len(l[0])):
                temp = []
                for j in xrange(self.batch_size):
                    temp.append(l[j][i])
                output.append(temp)
            return output

        batch_item_inputs = batch_major(item_inputs)

        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set[bucket_id]):
            finished = True

        return users, batch_item_inputs, positions, valids, finished

    def model_with_buckets(self, inputs, targets, weights,
                           buckets, cell, embeddingAttribute, dtype,
                           per_example_loss=False, name=None, devices=None):

        all_inputs = inputs + targets + weights
        losses = []
        losses_full = []
        outputs = []
        outputs_full = []
        topk_values = []
        topk_indexes = []

        def softmax_loss_function(x, y): return self.embeddingAttribute.compute_loss(
            x, y, loss=self.loss, device=devices[2])

        with tf.device(devices[1]):
            init_state = cell.zero_state(self.batch_size, dtype)

        with tf.name_scope(name, "model_with_buckets", all_inputs):
            # with ops.op_scope(all_inputs, name, "model_with_buckets"):
            for j, bucket in enumerate(buckets):
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=True if j > 0 else None):

                    with tf.device(devices[1]):
                        bucket_outputs, _ = tf.contrib.rnn.static_rnn(
                            cell, inputs[:bucket], initial_state=init_state)
                    with tf.device(devices[2]):

                        bucket_outputs_full = [self.embeddingAttribute.get_prediction(
                            x, device=devices[2], output_feat=self.output_feat) for x in bucket_outputs]

                        if self.loss in ['warp', 'ce']:
                            t = targets
                            bucket_outputs = [self.embeddingAttribute.get_prediction(
                                x, device=devices[2], output_feat=self.output_feat) for x in bucket_outputs]
                        elif self.loss in ['mw']:
                            # bucket_outputs0 = [self.embeddingAttribute.get_prediction(x, pool='sampled', device=devices[2]) for x in bucket_outputs]
                            t, bucket_outputs0 = [], []

                            for i in xrange(len(bucket_outputs)):
                                x = bucket_outputs[i]
                                ids = self.target_ids[i]
                                bucket_outputs0.append(self.embeddingAttribute.get_prediction(
                                    x, pool='sampled', device=devices[2], output_feat=self.output_feat))
                                t.append(self.embeddingAttribute.get_target_score(
                                    x, ids, device=devices[2]))
                            bucket_outputs = bucket_outputs0

                        outputs.append(bucket_outputs)
                        outputs_full.append(bucket_outputs_full)

                        if per_example_loss:
                            losses.append(sequence_loss_by_example(
                                outputs[-1], t[:bucket], weights[:bucket],
                                softmax_loss_function=softmax_loss_function))
                            losses_full.append(sequence_loss_by_example(
                                outputs_full[-1], t[:bucket], weights[:bucket],
                                softmax_loss_function=softmax_loss_function))
                        else:
                            losses.append(sequence_loss(
                                outputs[-1], t[:bucket], weights[:bucket],
                                softmax_loss_function=softmax_loss_function))
                            losses_full.append(sequence_loss(
                                outputs_full[-1], t[:bucket], weights[:bucket], softmax_loss_function=softmax_loss_function))
                        topk_value, topk_index = [], []

                        for full_logits in outputs_full[-1]:
                            value, index = tf.nn.top_k(tf.nn.softmax(
                                full_logits), self.topk_n, sorted=True)
                            topk_value.append(value)
                            topk_index.append(index)
                        topk_values.append(topk_value)
                        topk_indexes.append(topk_index)

        return outputs, losses, outputs_full, losses_full, topk_values, topk_indexes


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, default: "sequence_loss_by_example".

    Returns:
      1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with tf.name_scope(name, "sequence_loss_by_example", logits + targets + weights):
        # with ops.op_scope(logits + targets + weights,name, "sequence_loss_by_example"):
        log_perp_list = []
        for logit, target, weight in zip(logits, targets, weights):
            if softmax_loss_function is None:
                # TODO(irving,ebrevdo): This reshape is needed because
                # sequence_loss_by_example is called with scalars sometimes, which
                # violates our general scalar strictness policy.
                target = array_ops.reshape(target, [-1])
                crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    logit, target)
            else:
                crossent = softmax_loss_function(logit, target)
            log_perp_list.append(crossent * weight)

        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            total_size = math_ops.add_n(weights)
            # Just to avoid division by 0 for all-0 weights.
            total_size += 1e-12
            log_perps /= total_size
    return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=False,
                  softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      average_across_batch: If set, divide the returned cost by the batch size.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
      A scalar float Tensor: The average log-perplexity per symbol (weighted).

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """

    with tf.name_scope(name, "sequence_loss", logits + targets + weights):
        # with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
        cost = math_ops.reduce_sum(sequence_loss_by_example(
            logits, targets, weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            total_size = tf.reduce_sum(tf.sign(weights[0]))
            return cost / math_ops.cast(total_size, cost.dtype)
        else:
            return cost
