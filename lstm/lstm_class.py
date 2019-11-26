from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import logging

from seqModel import SeqModel

# import pandas as pd
# import configparser
# import env

sys.path.insert(0, '../utils')
sys.path.insert(0, '../attributes')

import embed_attribute
from input_attribute import read_data as read_attributed_data


import data_iterator
from data_iterator import DataIterator
from best_buckets import *
from tensorflow.python.client import timeline
from prepare_train import positive_items, item_frequency, sample_items, to_week
from attrdict import AttrDict
from pandatools import pd
from load_config import load_configurations
from sqlalchemy import create_engine


class lstm(object):
    """ lstm class object used to recommend "entity2" to "entity1"
    using LSTM-based seq2seq model on an implicit rating dataset.
    """

    def __init__(self, config):
        """ Class constructor, initializes key components
        """
        # setup
        self.config = config
        self.config['db_config'] = load_configurations(
            'config/environment/' + self.config['environment'] + '.json')
        self.ent1 = self.config['entity1_ID']
        self.ent2 = self.config['entity2_ID']
        self.FLAGS = AttrDict()
        # self.config['db_config'] = load_configurations( 'config/environment/'+self.config['environment']+'.json' )
        # datasets, paths, and preprocessing
        self.FLAGS.dataset = self.config['dataset_name']
        self.FLAGS.raw_data = self.config['raw_data_dir']
        self.FLAGS.data_dir = self.config['cache_dir']
        self.FLAGS.train_dir = self.config['train_dir']
        self.FLAGS.test = self.config['test']
        self.FLAGS.combine_att = self.config['combine_att']
        self.FLAGS.use_item_feature = self.config['use_item_feature']
        self.FLAGS.use_user_feature = self.config['use_user_feature']
        self.FLAGS.item_vocab_size = self.config['item_vocab_size']
        self.FLAGS.vocab_min_thresh = self.config['vocab_min_thresh']

        # tuning hypers
        self.FLAGS.loss = self.config['loss']
        self.FLAGS.learning_rate = self.config['learning_rate']
        self.FLAGS.learning_rate_decay_factor = self.config['learning_rate_decay_factor']
        self.FLAGS.max_gradient_norm = self.config['max_gradient_norm']
        self.FLAGS.keep_prob = self.config['keep_prob']
        self.FLAGS.power = self.config['power']
        self.FLAGS.batch_size = self.config['batch_size']
        self.FLAGS.size = self.config['size']
        self.FLAGS.num_layers = self.config['num_layers']
        self.FLAGS.n_epoch = self.config['n_epoch']
        self.FLAGS.L = self.config['L']
        self.FLAGS.n_bucket = self.config['n_bucket']
        self.FLAGS.patience = self.config['patience']
        # tf.app.flags.DEFINE_integer("steps_per_checkpoint", self.config['steps_per_checkpoint'],"How many training steps to do per checkpoint.")

        # recommendation
        self.FLAGS.recommend = self.config['recommend']
        self.FLAGS.saverec = self.config['saverec']
        self.FLAGS.recommend_new = self.config['recommend_new']
        self.FLAGS.topk = self.config['topk']

        # for ensemble
        self.FLAGS.ensemble = self.config['ensemble']
        self.FLAGS.ensemble_suffix = self.config['ensemble_suffix']
        self.FLAGS.seed = self.config['seed']

        # attribute model variants
        self.FLAGS.output_feat = self.config['output_feat']
        self.FLAGS.use_sep_item = self.config['use_sep_item']
        self.FLAGS.no_input_item_feature = self.config['no_input_item_feature']
        self.FLAGS.use_concat = self.config['use_concat']
        self.FLAGS.no_user_id = self.config['no_user_id']

        # devices
        self.FLAGS.N = self.config['N']

        #
        self.FLAGS.withAdagrad = self.config['withAdagrad']
        self.FLAGS.fromScratch = self.config['fromScratch']
        self.FLAGS.saveCheckpoint = self.config['saveCheckpoint']
        self.FLAGS.profile = self.config['profile']

        # others...
        self.FLAGS.ta = self.config['ta']
        self.FLAGS.user_sample = self.config['user_sample']

        self.FLAGS.after40 = self.config['after40']
        self.FLAGS.split = self.config['split']

        self.FLAGS.n_sampled = self.config['n_sampled']
        self.FLAGS.n_resample = self.config['n_resample']

        # for beam_search
        self.FLAGS.beam_search = self.config['beam_search']
        self.FLAGS.beam_size = self.config['beam_size']

        self.FLAGS.max_train_data_size = self.config['max_train_data_size']
        self.FLAGS.old_att = self.config['old_att']

        _buckets = []

        if self.FLAGS.test:
            if self.FLAGS.data_dir[-1] == '/':
                self.FLAGS.data_dir = self.FLAGS.data_dir[:-1] + '_test'
            else:
                self.FLAGS.data_dir = self.FLAGS.data_dir + '_test'

        if not os.path.exists(self.FLAGS.train_dir):
            os.mkdir(self.FLAGS.train_dir)

        if self.FLAGS.beam_search:
            self.FLAGS.batch_size = 1
            self.FLAGS.n_bucket = 1

        return

    def mylog(self, msg):
        logger = logging.getLogger('main')
        logger.info(msg)

    def split_buckets(self, array, buckets):
        """
        array : [(user,[items])]
        return:
        d : [[(user, [items])]]
        """
        d = [[] for i in xrange(len(buckets))]
        for u, items in array:
            index = self.get_buckets_id(len(items), buckets)
            if index >= 0:
                d[index].append((u, items))
        return d

    def get_buckets_id(self, l, buckets):
        id = -1
        for i in xrange(len(buckets)):
            if l <= buckets[i]:
                id = i
                break
        return id

    def form_sequence_prediction(self, data, uids, maxlen, START_ID):
        """
        Args:
          data = [(user_id,[item_id])]
          uids = [user_id]
         Return:
          d : [(user_id,[item_id])]
        """
        d = []
        m = {}
        for uid, items in data:
            m[uid] = items
        for uid in uids:
            if uid in m:
                items = [START_ID] + m[uid][-(maxlen - 1):]
            else:
                items = [START_ID]
            d.append((uid, items))

        return d

    def form_sequence(self, data, maxlen=100):
        """
        Args:
          data = [(u,i,week)]
        Return:
          d : [(user_id, [item_id])]
        """

        users = []
        items = []
        d = {}  # d[u] = [(i,week)]
        for u, i, week in data:
            if not u in d:
                d[u] = []
            d[u].append((i, week))

        dd = []
        n_all_item = 0
        n_rest_item = 0
        for u in d:
            tmp = sorted(d[u], key=lambda x: x[1])
            n_all_item += len(tmp)
            while True:
                new_tmp = [x[0] for x in tmp][:maxlen]
                n_rest_item += len(new_tmp)
                # make sure every sequence has at least one item
                if len(new_tmp) > 0:
                    dd.append((u, new_tmp))
                if len(tmp) <= maxlen:
                    break
                else:
                    if len(tmp) - maxlen <= 7:
                        tmp = tmp[maxlen - 10:]
                    else:
                        tmp = tmp[maxlen:]

        # count below not valid any more
        # mylog("All item: {} Rest item: {} Remove item: {}".format(n_all_item, n_rest_item, n_all_item - n_rest_item))

        return dd

    def prepare_warp(self, embAttr, data_tr, data_va):
        pos_item_list, pos_item_list_val = {}, {}
        for t in data_tr:
            u, i_list = t
            pos_item_list[u] = list(set(i_list))
        for t in data_va:
            u, i_list = t
            pos_item_list_val[u] = list(set(i_list))
        embAttr.prepare_warp(pos_item_list, pos_item_list_val)

    def get_device_address(self, s):
        add = []
        if s == "":
            for i in xrange(3):
                add.append("/cpu:0")
        else:
            add = ["/gpu:{}".format(int(x)) for x in s]
        self.mylog(add)
        return add

    def split_train_dev(self, seq_all, ratio=0.05):
        random.seed(self.FLAGS.seed)
        seq_tr, seq_va = [], []
        for item in seq_all:
            r = random.random()
            if r < ratio:
                seq_va.append(item)
            else:
                seq_tr.append(item)
        return seq_tr, seq_va

    def get_data(self, raw_data, recommend=False):
        data_dir = self.FLAGS.data_dir
        combine_att = self.FLAGS.combine_att
        test = self.FLAGS.test
        logits_size_tr = self.FLAGS.item_vocab_size
        thresh = self.FLAGS.vocab_min_thresh
        use_user_feature = self.FLAGS.use_user_feature
        use_item_feature = self.FLAGS.use_item_feature
        no_user_id = self.FLAGS.no_user_id

        (data_tr, data_va, u_attr, i_attr, item_ind2logit_ind, logit_ind2item_ind,
            user_index, item_index) = read_attributed_data(
            raw_data_dir=raw_data,
            data_dir=data_dir,
            combine_att=combine_att,
            logits_size_tr=logits_size_tr,
            thresh=thresh,
            use_user_feature=use_user_feature,
            use_item_feature=use_item_feature,
            no_user_id=no_user_id,
            test=test,
            mylog=self.mylog,
            config=self.config)

        # remove unk
        data_tr = [p for p in data_tr if (p[1] in item_ind2logit_ind)]

        # remove items before week 40
        if self.FLAGS.after40:
            data_tr = [p for p in data_tr if (to_week(p[2]) >= 40)]

        # item frequency (for sampling)
        item_population, p_item = item_frequency(data_tr, self.FLAGS.power)

        # UNK and START
        # mylog(len(item_ind2logit_ind))
        # mylog(len(logit_ind2item_ind))
        # mylog(len(item_index))
        START_ID = len(item_index)
        # START_ID = i_attr.get_item_last_index()
        item_ind2logit_ind[START_ID] = 0
        seq_all = self.form_sequence(data_tr, maxlen=self.FLAGS.L)
        seq_tr0, seq_va0 = self.split_train_dev(seq_all, ratio=0.05)

        # calculate buckets
        global _buckets
        _buckets = calculate_buckets(
            seq_tr0 + seq_va0, self.FLAGS.L, self.FLAGS.n_bucket)
        _buckets = sorted(_buckets)

        # split_buckets
        seq_tr = self.split_buckets(seq_tr0, _buckets)
        seq_va = self.split_buckets(seq_va0, _buckets)

        # get test data
        if recommend:
            from evaluate import Evaluation as Evaluate
            evaluation = Evaluate(raw_data, test=test,
                                  config=self.config, mylog=self.mylog)
            uinds = evaluation.get_uinds()
            seq_test = self.form_sequence_prediction(
                seq_all, uinds, self.FLAGS.L, START_ID)
            _buckets = calculate_buckets(
                seq_test, self.FLAGS.L, self.FLAGS.n_bucket)
            _buckets = sorted(_buckets)
            seq_test = self.split_buckets(seq_test, _buckets)
        else:
            seq_test = []
            evaluation = None
            uinds = []

        # create embedAttr

        devices = self.get_device_address(self.FLAGS.N)
        with tf.device(devices[0]):
            u_attr.set_model_size(self.FLAGS.size)
            i_attr.set_model_size(self.FLAGS.size)

            embAttr = embed_attribute.EmbeddingAttribute(
                u_attr, i_attr, self.FLAGS.batch_size, self.FLAGS.n_sampled, _buckets[-1], self.FLAGS.use_sep_item, item_ind2logit_ind, logit_ind2item_ind, devices=devices)

            if self.FLAGS.loss in ["warp", 'mw']:
                self.prepare_warp(embAttr, seq_tr0, seq_va0)

        return seq_tr, seq_va, seq_test, embAttr, START_ID, item_population, p_item, evaluation, uinds, user_index, item_index, logit_ind2item_ind

    def create_model(self, session, embAttr, START_ID, run_options, run_metadata):
        devices = self.get_device_address(self.FLAGS.N)
        dtype = tf.float32
        model = SeqModel(_buckets,
                         self.FLAGS.size,
                         self.FLAGS.num_layers,
                         self.FLAGS.max_gradient_norm,
                         self.FLAGS.batch_size,
                         self.FLAGS.learning_rate,
                         self.FLAGS.learning_rate_decay_factor,
                         embAttr,
                         withAdagrad=self.FLAGS.withAdagrad,
                         num_samples=self.FLAGS.n_sampled,
                         dropoutRate=self.FLAGS.keep_prob,
                         START_ID=START_ID,
                         loss=self.FLAGS.loss,
                         dtype=dtype,
                         devices=devices,
                         use_concat=self.FLAGS.use_concat,
                         no_user_id=False,  # to remove this argument
                         output_feat=self.FLAGS.output_feat,
                         no_input_item_feature=self.FLAGS.no_input_item_feature,
                         topk_n=self.FLAGS.topk,
                         run_options=run_options,
                         run_metadata=run_metadata
                         )

        ckpt = tf.train.get_checkpoint_state(self.FLAGS.train_dir)
        # if FLAGS.recommend or (not FLAGS.fromScratch) and ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):

        if self.FLAGS.recommend or self.FLAGS.beam_search or self.FLAGS.ensemble or (not self.FLAGS.fromScratch) and ckpt:
            self.mylog("Reading model parameters from %s" %
                       ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            self.mylog("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
        return model

    def show_all_variables(self):
        all_vars = tf.global_variables()
        for var in all_vars:
            self.mylog(var.name)

    def train(self):
        raw_data = self.FLAGS.raw_data

        # Read Data
        self.mylog("Reading Data...")
        train_set, dev_set, test_set, embAttr, START_ID, item_population, p_item, _, _, _, _, _ = self.get_data(
            raw_data)
        n_targets_train = np.sum(
            [np.sum([len(items) for uid, items in x]) for x in train_set])
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) /
                               train_total_size for i in xrange(len(train_bucket_sizes))]
        dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(_buckets))]
        dev_total_size = int(sum(dev_bucket_sizes))

        # steps
        batch_size = self.FLAGS.batch_size
        n_epoch = self.FLAGS.n_epoch
        steps_per_epoch = int(train_total_size / batch_size)
        steps_per_dev = int(dev_total_size / batch_size)

        steps_per_checkpoint = int(steps_per_epoch / 2)
        total_steps = steps_per_epoch * n_epoch

        # reports
        self.mylog(_buckets)
        self.mylog("Train:")
        self.mylog("total: {}".format(train_total_size))
        self.mylog("bucket sizes: {}".format(train_bucket_sizes))
        self.mylog("Dev:")
        self.mylog("total: {}".format(dev_total_size))
        self.mylog("bucket sizes: {}".format(dev_bucket_sizes))
        self.mylog("")
        self.mylog("Steps_per_epoch: {}".format(steps_per_epoch))
        self.mylog("Total_steps:{}".format(total_steps))
        self.mylog("Steps_per_checkpoint: {}".format(steps_per_checkpoint))

        # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement = False, device_count={'CPU':8, 'GPU':1})) as sess:
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

            # runtime profile
            if self.FLAGS.profile:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None

            self.mylog("Creating Model.. (this can take a few minutes)")
            model = self.create_model(sess, embAttr, START_ID,
                                      run_options, run_metadata)
            self.show_all_variables()

            # Data Iterators
            dite = DataIterator(model, train_set, len(
                train_buckets_scale), batch_size, train_buckets_scale)

            iteType = 0
            if iteType == 0:
                self.mylog("withRandom")
                ite = dite.next_random()
            elif iteType == 1:
                self.mylog("withSequence")
                ite = dite.next_sequence()

            # statistics during training
            step_time, loss = 0.0, 0.0
            current_step = 0
            previous_losses = []
            his = []
            low_ppx = float("inf")
            low_ppx_step = 0
            steps_per_report = 30
            n_targets_report = 0
            report_time = 0
            n_valid_sents = 0
            patience = self.FLAGS.patience
            item_sampled, item_sampled_id2idx = None, None

            while current_step < total_steps:

                # start
                start_time = time.time()

                # re-sample every once a while
                if self.FLAGS.loss in ['mw', 'mce'] and current_step % self.FLAGS.n_resample == 0:
                    item_sampled, item_sampled_id2idx = sample_items(
                        item_population, self.FLAGS.n_sampled, p_item)
                else:
                    item_sampled = None

                # data and train
                users, inputs, outputs, weights, bucket_id = ite.next()

                L = model.step(sess, users, inputs, outputs, weights, bucket_id,
                               item_sampled=item_sampled, item_sampled_id2idx=item_sampled_id2idx)

                # loss and time
                step_time += (time.time() - start_time) / steps_per_checkpoint

                loss += L
                current_step += 1
                n_valid_sents += np.sum(np.sign(weights[0]))

                # for report
                report_time += (time.time() - start_time)
                n_targets_report += np.sum(weights)

                if current_step % steps_per_report == 0:
                    self.mylog("--------------------" + "Report" +
                               str(current_step) + "-------------------")
                    self.mylog("StepTime: {} Speed: {} targets / sec in total {} targets".format(
                        report_time / steps_per_report, n_targets_report * 1.0 / report_time, n_targets_train))

                    report_time = 0
                    n_targets_report = 0

                    # Create the Timeline object, and write it to a json
                    if self.FLAGS.profile:
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        with open('timeline.json', 'w') as f:
                            f.write(ctf)
                        exit()

                if current_step % steps_per_checkpoint == 0:
                    self.mylog("--------------------" + "TRAIN" +
                               str(current_step) + "-------------------")
                    # Print statistics for the previous epoch.

                    loss = loss / n_valid_sents
                    perplexity = math.exp(
                        float(loss)) if loss < 300 else float("inf")
                    self.mylog("global step %d learning rate %.4f step-time %.2f perplexity " "%.2f" %
                               (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

                    train_ppx = perplexity

                    # Save checkpoint and zero timer and loss.
                    step_time, loss, n_valid_sents = 0.0, 0.0, 0

                    # dev data
                    self.mylog("--------------------" + "DEV" +
                               str(current_step) + "-------------------")
                    eval_loss, eval_ppx = self.evaluate(
                        sess, model, dev_set, item_sampled_id2idx=item_sampled_id2idx)
                    self.mylog("dev: ppx: {}".format(eval_ppx))

                    his.append([current_step, train_ppx, eval_ppx])

                    if eval_ppx < low_ppx:
                        patience = self.FLAGS.patience
                        low_ppx = eval_ppx
                        low_ppx_step = current_step
                        checkpoint_path = os.path.join(
                            self.FLAGS.train_dir, "best.ckpt")
                        self.mylog("Saving best model....")
                        s = time.time()
                        model.saver.save(sess, checkpoint_path,
                                         global_step=0, write_meta_graph=False)
                        self.mylog("Best model saved using {} sec".format(
                            time.time() - s))
                    else:
                        patience -= 1

                    if patience <= 0:
                        self.mylog(
                            "Training finished. Running out of patience.")
                        break

                    sys.stdout.flush()

    def evaluate(self, sess, model, data_set, item_sampled_id2idx=None):
        # Run evals on development set and print their perplexity/loss.
        dropoutRateRaw = self.FLAGS.keep_prob
        sess.run(model.dropout10_op)

        start_id = 0
        loss = 0.0
        n_steps = 0
        n_valids = 0
        batch_size = self.FLAGS.batch_size

        dite = DataIterator(model, data_set, len(_buckets), batch_size, None)
        ite = dite.next_sequence(stop=True)

        for users, inputs, outputs, weights, bucket_id in ite:
            L = model.step(sess, users, inputs, outputs,
                           weights, bucket_id, forward_only=True)
            loss += L
            n_steps += 1
            n_valids += np.sum(np.sign(weights[0]))

        loss = loss / (n_valids)
        ppx = math.exp(loss) if loss < 300 else float("inf")

        sess.run(model.dropoutAssign_op)

        return loss, ppx

    def recommend(self):
        raw_data = self.FLAGS.raw_data

        # Read Data
        self.mylog("recommend")
        self.mylog("Reading Data...")
        _, _, test_set, embAttr, START_ID, _, _, evaluation, uinds, user_index, item_index, logit_ind2item_ind = self.get_data(
            raw_data, recommend=True)
        test_bucket_sizes = [len(test_set[b]) for b in xrange(len(_buckets))]
        test_total_size = int(sum(test_bucket_sizes))

        # reports
        self.mylog(_buckets)
        self.mylog("Test:")
        self.mylog("total: {}".format(test_total_size))
        self.mylog("buckets: {}".format(test_bucket_sizes))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

            # runtime profile
            if self.FLAGS.profile:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None

            self.mylog("Creating Model")
            model = self.create_model(sess, embAttr, START_ID,
                                      run_options, run_metadata)
            self.show_all_variables()

            sess.run(model.dropoutRate.assign(1.0))

            start_id = 0
            n_steps = 0
            batch_size = self.FLAGS.batch_size

            dite = DataIterator(model, test_set, len(
                _buckets), batch_size, None)
            ite = dite.next_sequence(stop=True, recommend=True)

            n_total_user = len(uinds)
            n_recommended = 0
            uind2rank = {}
            for r, uind in enumerate(uinds):
                uind2rank[uind] = r
            rec = np.zeros((n_total_user, self.FLAGS.topk), dtype=int)
            rec_value = np.zeros((n_total_user, self.FLAGS.topk), dtype=float)
            start = time.time()

            for users, inputs, positions, valids, bucket_id in ite:
                results = model.step_recommend(
                    sess, users, inputs, positions, bucket_id)
                for i, valid in enumerate(valids):
                    if valid == 1:
                        n_recommended += 1
                        if n_recommended % 1000 == 0:
                            self.mylog("Evaluating n {} bucket_id {}".format(
                                n_recommended, bucket_id))
                        uind, topk_values, topk_indexes = results[i]
                        rank = uind2rank[uind]
                        rec[rank, :] = topk_indexes
                        rec_value[rank, :] = topk_values
                n_steps += 1
            end = time.time()
            self.mylog("Time used {} sec for {} steps {} users ".format(
                end - start, n_steps, n_recommended))

            ind2id = {}
            for iid in item_index:
                iind = item_index[iid]
                assert(iind not in ind2id)
                ind2id[iind] = iid

            uind2id = {}
            for uid in user_index:
                uind = user_index[uid]
                assert(uind not in uind2id)
                uind2id[uind] = uid

            R = {}
            for i in xrange(n_total_user):
                uid = uind2id[uinds[i]]
                R[uid] = [ind2id[logit_ind2item_ind[v]]
                          for v in list(rec[i, :])]

            self.recs = pd.DataFrame.from_dict(R, orient="index")
            self.recs = self.recs.stack().reset_index()
            self.recs.columns = [self.ent1, 'slot', self.ent2]

            evaluation.eval_on(R)

            scores_self, scores_ex = evaluation.get_scores()
            self.mylog(
                "====evaluation scores (NDCG, RECALL, PRECISION, MAP) @ 2,5,10,20,30====")
            self.mylog("METRIC_FORMAT (self): {}".format(scores_self))
            self.mylog("METRIC_FORMAT (ex  ): {}".format(scores_ex))

            if self.FLAGS.saverec:
                self.write_to_db(self.recs)

    def ensemble(self):
        raw_data = self.FLAGS.raw_data
        # Read Data
        self.mylog("Ensemble {} {}".format(
            self.FLAGS.train_dir, self.FLAGS.ensemble_suffix))
        self.mylog("Reading Data...")
        # task = Task(FLAGS.dataset)
        _, _, test_set, embAttr, START_ID, _, _, evaluation, uinds, user_index, item_index, logit_ind2item_ind = self.get_data(
            raw_data, recommend=True)
        test_bucket_sizes = [len(test_set[b]) for b in xrange(len(_buckets))]
        test_total_size = int(sum(test_bucket_sizes))

        # reports
        self.mylog(_buckets)
        self.mylog("Test:")
        self.mylog("total: {}".format(test_total_size))
        self.mylog("buckets: {}".format(test_bucket_sizes))

        # load top_index, and top_value
        suffixes = self.FLAGS.ensemble_suffix.split(',')
        top_indexes = []
        top_values = []
        for suffix in suffixes:
            # dir_path = FLAGS.train_dir+suffix
            dir_path = self.FLAGS.train_dir.replace('seed', 'seed' + suffix)
            self.mylog("Loading results from {}".format(dir_path))
            index_path = os.path.join(
                dir_path, "top{}_index.npy".format(self.FLAGS.topk))
            value_path = os.path.join(
                dir_path, "top{}_value.npy".format(self.FLAGS.topk))
            top_index = np.load(index_path)
            top_value = np.load(value_path)
            top_indexes.append(top_index)
            top_values.append(top_value)

        # ensemble
        rec = np.zeros(top_indexes[0].shape)
        for row in xrange(rec.shape[0]):
            v = {}
            for i in xrange(len(suffixes)):
                for col in xrange(rec.shape[1]):
                    index = top_indexes[i][row, col]
                    value = top_values[i][row, col]
                    if index not in v:
                        v[index] = 0
                    v[index] += value
            items = [(index, v[index] / len(suffixes)) for index in v]
            items = sorted(items, key=lambda x: -x[1])
            rec[row:] = [x[0] for x in items][:FLAGS.topk]
            if row % 1000 == 0:
                self.mylog("Ensembling n {}".format(row))

        ind2id = {}
        for iid in item_index:
            uind = item_index[iid]
            assert(uind not in ind2id)
            ind2id[uind] = iid

        uind2id = {}
        for uid in user_index:
            uind = user_index[uid]
            assert(uind not in uind2id)
            uind2id[uind] = uid

        R = {}
        for i in xrange(n_total_user):
            uind = uinds[i]
            uid = uind2id[uind]
            R[uid] = [ind2id[logit_ind2item_ind[v]] for v in list(rec[i, :])]

        evaluation.eval_on(R)

        scores_self, scores_ex = evaluation.get_scores()
        self.mylog(
            "====evaluation scores (NDCG, RECALL, PRECISION, MAP) @ 2,5,10,20,30====")
        self.mylog("METRIC_FORMAT (self): {}".format(scores_self))
        self.mylog("METRIC_FORMAT (ex  ): {}".format(scores_ex))

    def beam_search(self):
        self.mylog("Reading Data...")
        task = Task(self.FLAGS.dataset)
        _, _, test_set, embAttr, START_ID, _, _, evaluation, uids = read_data(
            task, test=True)
        test_bucket_sizes = [len(test_set[b]) for b in xrange(len(_buckets))]
        test_total_size = int(sum(test_bucket_sizes))

        # reports
        self.mylog(_buckets)
        self.mylog("Test:")
        self.mylog("total: {}".format(test_total_size))
        self.mylog("buckets: {}".format(test_bucket_sizes))

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:

            # runtime profile
            if self.FLAGS.profile:
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None

            self.mylog("Creating Model")
            model = self.create_model(sess, embAttr, START_ID,
                                      run_options, run_metadata)
            self.show_all_variables()
            model.init_beam_decoder()

            sess.run(model.dropoutRate.assign(1.0))

            start_id = 0
            n_steps = 0
            batch_size = self.FLAGS.batch_size

            dite = DataIterator(model, test_set, len(
                _buckets), batch_size, None)
            ite = dite.next_sequence(stop=True, recommend=True)

            n_total_user = len(uids)
            n_recommended = 0
            uid2rank = {}
            for r, uid in enumerate(uids):
                uid2rank[uid] = r
            rec = np.zeros((n_total_user, self.FLAGS.topk), dtype=int)
            rec_value = np.zeros((n_total_user, self.FLAGS.topk), dtype=float)
            start = time.time()

            for users, inputs, positions, valids, bucket_id in ite:
                self.mylog(inputs)
                self.mylog(positions)
                results = model.beam_step(
                    sess, index=0, user_input=users, item_inputs=inputs, sequence_length=positions, bucket_id=bucket_id)
                break

    def write_to_db(self, df):
        """ write recommenations to database
        """

        # form the connection string used to connect to recommendation_service DB
        try:
            cxn_string = "mysql+mysqlconnector://%(user)s:%(password)s@%(host)s/%(database)s" % \
                         self.config['db_config']['recommendation_service']
            engine = create_engine(cxn_string)
        except:
            self.mylog(
                'error creating connection engine, check connection string: %s' % cxn_string)
            self.mylog(sys.exc_info())
            sys.exit()

        # write new records to target output DB table
        try:
            start = time.time()
            self.mylog('writing new recommendations to db')

            # decode entity1 and entity2 back into alpha-numeric IDs
            #df[self.ent1] = self.ent1_encoder.inverse_transform(df[self.ent1])
            #df[self.ent2] = self.ent2_encoder.inverse_transform(df[self.ent2])
            # add timestamp to mark when the recommendations were made
            df['createdAt'] = time.strftime('%Y-%m-%d %H:%M:%S')
            df = df[['createdAt', self.ent1, self.ent2, 'slot']]

            # write to output table 5000 rows at a time; note that column "rank" is ommitted
            df[[self.ent1, self.ent2, 'createdAt']].to_sql(index=False,
                                                           name=self.config['rec_output_config']['output_table'],
                                                           con=engine,
                                                           if_exists='append',
                                                           chunksize=5000)
            self.mylog('writing new recommendations took %.2f seconds.' %
                       (time.time() - start))

        except:
            self.mylog('failed writing new recommendations to \"recommendation_service.%s\"' %
                       self.config['rec_output_config']['output_table'])
            self.mylog(sys.exc_info())
            sys.exit()

        # clear up old recommendations
        try:
            start = time.time()
            self.mylog('clearing up previous recommendations')

            engine.execute('delete from %(table)s where createdAt < \'%(timestamp)s\';' %
                           {'table': self.config['rec_output_config']['output_table'],
                            'timestamp': df.createdAt.values.tolist()[0]})
            self.mylog('clearing up old recommendations took %.2f seconds.' % (
                time.time() - start))

        except:
            self.mylog('failed clearing up old recommendations before %s from %s' % (df.createdAt.values.tolist()[0],
                                                                                     self.config['rec_output_config']['output_table']))
            self.mylog(sys.exc_info())
            sys.exit()

        return

    def get_past_rec(self):
        """ Fetches past recommendations from the database
        """

        # form the connection string used to connect to recommendation_service DB
        try:
            cxn_string = "mysql+mysqlconnector://%(user)s:%(password)s@%(host)s/%(database)s" % \
                         self.config['db_config']['recommendation_service']
            engine = create_engine(cxn_string)
        except:
            self.mylog(
                'error creating connection engine, check connection string: %s' % cxn_string)
            self.mylog(sys.exc_info())
            sys.exit()

        # backup the current recommendations in recommendation_service
        # this moves current recommendations to the archive table
        try:
            self.mylog('backing up current recommendations to archive')
            start = time.time()
            query = 'SELECT * FROM %s' % self.config['rec_output_config']['output_table']
            data_iterator = pd.read_sql_query(query, engine, chunksize=5000)
            for records in data_iterator:
                # copy records to archive
                records[['createdAt', self.ent1, self.ent2]].to_sql(index=False,
                                                                    name=self.config['rec_output_config']['archive_table'],
                                                                    con=engine,
                                                                    if_exists='append')
        except:
            self.mylog('failed backing up old recommendations')
            self.mylog(sys.exc_info())
            sys.exit()

        # get list of recommendations in past N days from archive table
        try:
            self.mylog('fetching past recommendations')
            start = time.time()
            query = """SELECT %(entity_one)s, %(entity_two)s
					   FROM %(table)s
					   WHERE createdAt >= DATE_SUB(CURDATE(),interval %(recs_past_N_days)s day)"""\
                               % {'entity_one': self.ent1_view.columns[0],
                                  'entity_two': self.ent2_view.columns[0],
                                  'table': self.config['rec_output_config']['archive_table'],
                                  'recs_past_N_days': self.config['recs_past_N_days']
                                  }
            self.past_rec = pd.read_sql_query(query, cxn_string)
            self.mylog('fetching past recommendations took %.2f seconds' % (
                time.time() - start))
        except:
            self.mylog('error executing query to fetch past recommendations')
            self.mylog(sys.exc_info())
            sys.exit()

        # filter out ent1 and ent2 IDs in past rec that are not present in the current dataset
        # (this is more of a safety net mechanism)
        # self.past_rec = self.past_rec[self.past_rec[self.ent1].isin(self.ent1_encoder.classes_.tolist())]
        # self.past_rec = self.past_rec[self.past_rec[self.ent2].isin(self.ent2_encoder.classes_.tolist())]
        # encode the ent1 and ent2 IDs from past rec
        # self.past_rec[self.ent1] = self.ent1_encoder.transform(self.past_rec[self.ent1])
        # self.past_rec[self.ent2] = self.ent2_encoder.transform(self.past_rec[self.ent2])

        return
