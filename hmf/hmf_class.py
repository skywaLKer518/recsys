from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import random
import time
import logging
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

sys.path.insert(0, '../utils')
sys.path.insert(0, '../attributes')

from input_attribute import read_data
from prepare_train import positive_items, item_frequency, sample_items
from attrdict import AttrDict
from pandatools import pd
from load_config import load_configurations
from sqlalchemy import create_engine


class hmf(object):
    """ hmf class object used to recommend "entity2" to "entity1"
    using Hybrid matrix factorization model (with deep layer extensions) on an implicit rating dataset.
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
        self.FLAGS.user_vocab_size = self.config['user_vocab_size']
        self.FLAGS.item_vocab_size = self.config['item_vocab_size']
        self.FLAGS.vocab_min_thresh = self.config['vocab_min_thresh']

        # tuning hypers
        self.FLAGS.loss = self.config['loss']
        self.FLAGS.loss_func = self.config['loss_func']
        self.FLAGS.loss_exp_p = self.config['loss_exp_p']
        self.FLAGS.learning_rate = self.config['learning_rate']
        self.FLAGS.keep_prob = self.config['keep_prob']
        self.FLAGS.learning_rate_decay_factor = self.config['learning_rate_decay_factor']
        self.FLAGS.batch_size = self.config['batch_size']
        self.FLAGS.size = self.config['size']
        self.FLAGS.patience = self.config['patience']
        self.FLAGS.n_epoch = self.config['n_epoch']
        self.FLAGS.steps_per_checkpoint = self.config['steps_per_checkpoint']

        # to recommend
        self.FLAGS.recommend = self.config['recommend']
        self.FLAGS.saverec = self.config['saverec']
        self.FLAGS.top_N_items = self.config['top_N_items']
        self.FLAGS.recommend_new = self.config['recommend_new']

        # nonlinear
        self.FLAGS.nonlinear = self.config['nonlinear']
        self.FLAGS.hidden_size = self.config['hidden_size']
        self.FLAGS.num_layers = self.config['num_layers']

        # algorithms with sampling
        self.FLAGS.power = self.config['power']
        self.FLAGS.n_resample = self.config['n_resample']
        self.FLAGS.n_sampled = self.config['n_sampled']

        self.FLAGS.sample_type = self.config['sample_type']
        self.FLAGS.user_sample = self.config['user_sample']
        self.FLAGS.seed = self.config['seed']

        #
        self.FLAGS.gpu = self.config['gpu']
        self.FLAGS.profile = self.config['profile']
        self.FLAGS.device_log = self.config['device_log']
        self.FLAGS.eval = self.config['eval']
        self.FLAGS.use_more_train = self.config['use_more_train']
        self.FLAGS.model_option = self.config['model_option']

        # self.FLAGS.max_train_data_size = self.config['max_train_data_size']
        # Xing related
        # self.FLAGS.ta = self.config['ta']

        if self.FLAGS.test:
            if self.FLAGS.data_dir[-1] == '/':
                self.FLAGS.data_dir = self.FLAGS.data_dir[:-1] + '_test'
            else:
                self.FLAGS.data_dir = self.FLAGS.data_dir + '_test'

        if not os.path.exists(self.FLAGS.train_dir):
            os.mkdir(self.FLAGS.train_dir)

        return

    def mylog(self, msg):
        logger = logging.getLogger('main')
        logger.info(msg)

        return

    def create_model(self, session, u_attributes=None, i_attributes=None,
                     item_ind2logit_ind=None, logit_ind2item_ind=None, logit_size_test=None, ind_item=None):
        loss = self.FLAGS.loss
        gpu = None if self.FLAGS.gpu == -1 else self.FLAGS.gpu
        n_sampled = self.FLAGS.n_sampled if self.FLAGS.loss in [
            'mw', 'mce'] else None
        import hmf_model
        model = hmf_model.LatentProductModel(self.FLAGS.user_vocab_size,
                                             self.FLAGS.item_vocab_size, self.FLAGS.size, self.FLAGS.num_layers,
                                             self.FLAGS.batch_size, self.FLAGS.learning_rate,
                                             self.FLAGS.learning_rate_decay_factor, u_attributes, i_attributes,
                                             item_ind2logit_ind, logit_ind2item_ind, loss_function=loss, GPU=gpu,
                                             logit_size_test=logit_size_test, nonlinear=self.FLAGS.nonlinear,
                                             dropout=self.FLAGS.keep_prob, n_sampled=n_sampled, indices_item=ind_item,
                                             top_N_items=self.FLAGS.top_N_items, hidden_size=self.FLAGS.hidden_size,
                                             loss_func=self.FLAGS.loss_func, loss_exp_p=self.FLAGS.loss_exp_p)

        if not os.path.isdir(self.FLAGS.train_dir):
            os.mkdir(self.FLAGS.train_dir)

        ckpt = tf.train.get_checkpoint_state(self.FLAGS.train_dir)

        if ckpt:
            self.mylog("Reading model parameters from %s" %
                       ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            self.mylog("Created model with fresh parameters.")
            # session.run(tf.global_variables_initializer())
            session.run(tf.global_variables_initializer())

        return model

    def train(self):
        raw_data = self.FLAGS.raw_data
        train_dir = self.FLAGS.train_dir
        data_dir = self.FLAGS.data_dir
        combine_att = self.FLAGS.combine_att
        test = self.FLAGS.test
        logits_size_tr = self.FLAGS.item_vocab_size
        thresh = self.FLAGS.vocab_min_thresh
        use_user_feature = self.FLAGS.use_user_feature
        use_item_feature = self.FLAGS.use_item_feature
        batch_size = self.FLAGS.batch_size
        steps_per_checkpoint = self.FLAGS.steps_per_checkpoint
        loss_func = self.FLAGS.loss
        max_patience = self.FLAGS.patience
        go_test = self.FLAGS.test
        max_epoch = self.FLAGS.n_epoch
        sample_type = self.FLAGS.sample_type
        power = self.FLAGS.power
        use_more_train = self.FLAGS.use_more_train
        profile = self.FLAGS.profile
        device_log = self.FLAGS.device_log

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=device_log)) as sess:
            run_options = None
            run_metadata = None

            if profile:
                # in order to profile
                from tensorflow.python.client import timeline
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                steps_per_checkpoint = 30

            self.mylog("reading data")
            (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind,
             logit_ind2item_ind, _, _) = read_data(
                raw_data_dir=raw_data,
                data_dir=data_dir,
                combine_att=combine_att,
                logits_size_tr=logits_size_tr,
                thresh=thresh,
                use_user_feature=use_user_feature,
                use_item_feature=use_item_feature,
                test=test,
                mylog=self.mylog)

            self.mylog("train/dev size: %d/%d" % (len(data_tr), len(data_va)))

            '''
			remove some rare items in both train and valid set
			this helps make train/valid set distribution similar
			to each other
			'''
            self.mylog("original train/dev size: %d/%d" %
                       (len(data_tr), len(data_va)))
            data_tr = [p for p in data_tr if (p[1] in item_ind2logit_ind)]
            data_va = [p for p in data_va if (p[1] in item_ind2logit_ind)]
            self.mylog("new train/dev size: %d/%d" %
                       (len(data_tr), len(data_va)))

            random.seed(self.FLAGS.seed)

            item_pop, p_item = item_frequency(data_tr, power)

            if use_more_train:
                item_population = range(len(item_ind2logit_ind))
            else:
                item_population = item_pop

            model = self.create_model(sess, u_attributes, i_attributes, item_ind2logit_ind,
                                      logit_ind2item_ind, ind_item=item_population)

            pos_item_list, pos_item_list_val = None, None

            if loss_func in ['warp', 'mw', 'rs', 'rs-sig', 'rs-sig2', 'bbpr']:
                pos_item_list, pos_item_list_val = positive_items(
                    data_tr, data_va)
                model.prepare_warp(pos_item_list, pos_item_list_val)

            self.mylog('started training')
            step_time, loss, current_step, auc = 0.0, 0.0, 0, 0.0

            repeat = 5 if loss_func.startswith('bpr') else 1
            patience = max_patience

            if os.path.isfile(os.path.join(train_dir, 'auc_train.npy')):
                auc_train = list(
                    np.load(os.path.join(train_dir, 'auc_train.npy')))
                auc_dev = list(np.load(os.path.join(train_dir, 'auc_dev.npy')))
                previous_losses = list(np.load(os.path.join(train_dir,
                                                            'loss_train.npy')))
                losses_dev = list(
                    np.load(os.path.join(train_dir, 'loss_dev.npy')))
                best_auc = max(auc_dev)
                best_loss = min(losses_dev)
            else:
                previous_losses, auc_train, auc_dev, losses_dev = [], [], [], []
                best_auc, best_loss = -1, 1000000

            item_sampled, item_sampled_id2idx = None, None

            if sample_type == 'random':
                get_next_batch = model.get_batch
            elif sample_type == 'permute':
                get_next_batch = model.get_permuted_batch
            else:
                print('not implemented!')
                exit()

            train_total_size = float(len(data_tr))
            n_epoch = max_epoch
            steps_per_epoch = int(1.0 * train_total_size / batch_size)
            total_steps = steps_per_epoch * n_epoch

            self.mylog("Train:")
            self.mylog("total: {}".format(train_total_size))
            self.mylog("Steps_per_epoch: {}".format(steps_per_epoch))
            self.mylog("Total_steps:{}".format(total_steps))
            self.mylog("Dev:")
            self.mylog("total: {}".format(len(data_va)))

            self.mylog("\n\ntraining start!")

            while True:
                ranndom_number_01 = np.random.random_sample()
                start_time = time.time()
                (user_input, item_input, neg_item_input) = get_next_batch(data_tr)

                if loss_func in ['mw', 'mce'] and current_step % self.FLAGS.n_resample == 0:
                    item_sampled, item_sampled_id2idx = sample_items(
                        item_population, self.FLAGS.n_sampled, p_item)
                else:
                    item_sampled = None

                step_loss = model.step(sess, user_input, item_input,
                                       neg_item_input, item_sampled, item_sampled_id2idx, loss=loss_func,
                                       run_op=run_options, run_meta=run_metadata)

                step_time += (time.time() - start_time) / \
                    steps_per_checkpoint
                loss += step_loss / steps_per_checkpoint
                current_step += 1

                if model.global_step.eval() > total_steps:
                    self.mylog(
                        "Training reaches maximum steps. Terminating...")
                    break

                if current_step % steps_per_checkpoint == 0:

                    if loss_func in ['ce', 'mce']:
                        perplexity = math.exp(
                            loss) if loss < 300 else float('inf')
                        self.mylog("global step %d learning rate %.4f step-time %.4f perplexity %.2f" %
                                   (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                    else:
                        self.mylog("global step %d learning rate %.4f step-time %.4f loss %.3f" %
                                   (model.global_step.eval(), model.learning_rate.eval(), step_time, loss))
                    if profile:
                        # Create the Timeline object, and write it to a json
                        tl = timeline.Timeline(run_metadata.step_stats)
                        ctf = tl.generate_chrome_trace_format()
                        with open('timeline.json', 'w') as f:
                            f.write(ctf)
                        exit()

                    # Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)

                    previous_losses.append(loss)
                    auc_train.append(auc)

                    # Reset timer and loss.
                    step_time, loss, auc = 0.0, 0.0, 0.0

                    if not self.FLAGS.eval:
                        continue

                    # Run evals on development set and print their loss.
                    l_va = len(data_va)
                    eval_loss, eval_auc = 0.0, 0.0
                    count_va = 0
                    start_time = time.time()
                    for idx_s in range(0, l_va, batch_size):
                        idx_e = idx_s + batch_size
                        if idx_e > l_va:
                            break
                        lt = data_va[idx_s:idx_e]
                        user_va = [x[0] for x in lt]
                        item_va = [x[1] for x in lt]
                        for _ in range(repeat):
                            item_va_neg = None
                            the_loss = 'warp' if loss_func == 'mw' else loss_func
                            eval_loss0 = model.step(sess, user_va, item_va, item_va_neg,
                                                    None, None, forward_only=True,
                                                    loss=the_loss)
                            eval_loss += eval_loss0
                            count_va += 1

                    eval_loss /= count_va
                    eval_auc /= count_va
                    step_time = (
                        time.time() - start_time) / count_va
                    if loss_func in ['ce', 'mce']:
                        eval_ppx = math.exp(
                            eval_loss) if eval_loss < 300 else float('inf')
                        self.mylog("  dev: perplexity %.2f eval_auc(not computed) %.4f step-time %.4f" % (
                            eval_ppx, eval_auc, step_time))
                    else:
                        self.mylog("  dev: loss %.3f eval_auc(not computed) %.4f step-time %.4f" % (eval_loss,
                                                                                                    eval_auc, step_time))

                    if eval_loss < best_loss and not go_test:
                        best_loss = eval_loss
                        patience = max_patience
                        checkpoint_path = os.path.join(
                            train_dir, "best.ckpt")
                        self.mylog('Saving best model...')
                        model.saver.save(sess, checkpoint_path,
                                         global_step=0, write_meta_graph=False)

                    if go_test:
                        checkpoint_path = os.path.join(
                            train_dir, "best.ckpt")
                        self.mylog('Saving best model...')
                        model.saver.save(sess, checkpoint_path,
                                         global_step=0, write_meta_graph=False)

                    if eval_loss > best_loss:
                        patience -= 1

                    auc_dev.append(eval_auc)
                    losses_dev.append(eval_loss)

                    if patience < 0 and not go_test:
                        self.mylog(
                            "no improvement for too long.. terminating..")
                        self.mylog("best loss %.4f" % best_loss)
                        break
        return

    def recommend(self, target_uids=[]):
        raw_data = self.FLAGS.raw_data
        data_dir = self.FLAGS.data_dir
        combine_att = self.FLAGS.combine_att
        logits_size_tr = self.FLAGS.item_vocab_size
        item_vocab_min_thresh = self.FLAGS.vocab_min_thresh
        use_user_feature = self.FLAGS.use_user_feature
        use_item_feature = self.FLAGS.use_item_feature
        batch_size = self.FLAGS.batch_size
        loss = self.FLAGS.loss
        top_n = self.FLAGS.top_N_items
        test = self.FLAGS.test
        device_log = self.FLAGS.device_log

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=device_log)) as sess:
            self.mylog("reading data")
            (_, _, u_attributes, i_attributes, item_ind2logit_ind, logit_ind2item_ind, user_index, item_index) = read_data(
                raw_data_dir=raw_data,
                data_dir=data_dir,
                combine_att=combine_att,
                logits_size_tr=logits_size_tr,
                thresh=item_vocab_min_thresh,
                use_user_feature=use_user_feature,
                use_item_feature=use_item_feature,
                test=test,
                mylog=self.mylog)

            model = self.create_model(sess, u_attributes, i_attributes,
                                      item_ind2logit_ind, logit_ind2item_ind, ind_item=None)

            Uinds = [user_index[v] for v in target_uids]

            N = len(Uinds)
            self.mylog("%d target users to recommend" % N)
            rec = np.zeros((N, top_n), dtype=int)

            count = 0
            time_start = time.time()
            for idx_s in range(0, N, batch_size):
                count += 1
                if count % 100 == 0:
                    self.mylog("idx: %d, c: %d" % (idx_s, count))

                idx_e = idx_s + batch_size
                if idx_e <= N:
                    users = Uinds[idx_s: idx_e]
                    recs = model.step(sess, users, None, None, forward_only=True,
                                      recommend=True)
                    rec[idx_s:idx_e, :] = recs
                else:
                    users = range(idx_s, N) + [0] * (idx_e - N)
                    users = [Uinds[t] for t in users]
                    recs = model.step(sess, users, None, None, forward_only=True,
                                      recommend=True)
                    idx_e = N
                    rec[idx_s:idx_e, :] = recs[:(idx_e - idx_s), :]

            time_end = time.time()
            self.mylog("Time used %.1f" % (time_end - time_start))

            # transform result to a dictionary
            # R[user_id] = [item_id1, item_id2, ...]

            ind2id = {}
            for iid in item_index:
                uind = item_index[iid]
                assert(uind not in ind2id)
                ind2id[uind] = iid
            R = {}
            for i in xrange(N):
                uid = target_uids[i]
                R[uid] = [ind2id[logit_ind2item_ind[v]]
                          for v in list(rec[i, :])]

            self.recs = pd.DataFrame.from_dict(R, orient="index")
            self.recs = self.recs.stack().reset_index()
            self.recs.columns = [self.ent1, 'slot', self.ent2]

        return R

    def compute_scores(self):
        raw_data = self.FLAGS.raw_data
        data_dir = self.FLAGS.data_dir
        dataset = self.FLAGS.dataset
        save_recommendation = self.FLAGS.saverec
        train_dir = self.FLAGS.train_dir
        test = self.FLAGS.test

        from evaluate import Evaluation as Evaluate
        evaluation = Evaluate(raw_data, test=test)

        # if filter:
        # get past N days of recommendations, these will be filtered out
        # from the final list to avoid making duplicate recommendations
        # self.mylog('backing up most recent recommendations & fetch recommendations in the past %d days' %self.config['recs_past_N_days'])
        # self.get_past_rec()

        R = self.recommend(evaluation.get_uids())

        evaluation.eval_on(R)
        scores_self, scores_ex = evaluation.get_scores()
        self.mylog(
            "====evaluation scores (NDCG, RECALL, PRECISION, MAP) @ 2,5,10,20,30====")
        self.mylog("METRIC_FORMAT (self): {}".format(scores_self))
        self.mylog("METRIC_FORMAT (ex  ): {}".format(scores_ex))

        if self.FLAGS.saverec:
            self.write_to_db(self.recs)

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
