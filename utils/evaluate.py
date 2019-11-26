from eval_metrics import metrics
from submit import load_submit, combine_sub, format_submit
from load_data import load_users, load_items, load_interactions
from os.path import isfile, join


class Evaluation(object):

    def __init__(self, raw_data_dir, test=False, config=None, mylog=None):

        res_filename = 'res_T_test.csv' if test else 'res_T.csv'
        if not isfile(join(raw_data_dir, res_filename)):
            print('eval file does not exist. creating ... ')
            self.create_eval_file(raw_data_dir, config, mylog)
        self.T = load_submit(res_filename, submit_dir=raw_data_dir)
        hist_filename = 'historical_train_test.csv' if test else 'historical_train.csv'
        self.hist = load_submit(hist_filename, submit_dir=raw_data_dir)

        self.Iatt, _, self.Iid2ind = load_items(raw_data_dir, config, mylog)
        self.Uatt, _, self.Uid2ind = load_users(raw_data_dir, config, mylog)

        self.Uids = self.get_uids()
        self.Uinds = [self.Uid2ind[v] for v in self.Uids]
        self.combine_sub = combine_sub
        return

    def get_user_n(self):
        return len(self.Uinds)

    def get_uids(self):
        return list(self.T.keys())

    def get_uinds(self):
        return self.Uinds

    def eval_on(self, rec):

        self.res = rec

        tmp_filename = 'rec'
        for k, v in rec.items():
            rec[k] = [str(v) for v in rec[k]]
            # if isinstance(R[k], list):

        #     R[k] = ','.join(str(xx) for xx in v)
        # for k, v in rec.items():
        #     rec[k] = v.split(',')

        r_ex = self.combine_sub(self.hist, rec, 1, users=self.Uatt)

        result = metrics(rec, self.T)
        l = result.values()
        self.s_self = [item for sublist in l for item in sublist]
        l = metrics(r_ex, self.T).values()
        self.s_ex = [item for sublist in l for item in sublist]
        return

    def get_scores(self):
        return self.s_self, self.s_ex

    def set_uinds(self, uinds):
        self.Uinds = uinds

    def create_eval_file(self, raw_data, config, mylog):
        DIR = raw_data
        interact, names = load_interactions(
            data_dir=DIR, config=config, mylog=mylog)

        interact_tr, interact_va, interact_te = interact

        data_tr = zip(list(interact_tr[:, 0]), list(
            interact_tr[:, 1]), list(interact_tr[:, 2]))
        data_va = zip(list(interact_va[:, 0]), list(
            interact_va[:, 1]), list(interact_va[:, 2]))
        data_te = zip(list(interact_te[:, 0]), list(
            interact_te[:, 1]), list(interact_te[:, 2]))

        seq_tr, seq_va, seq_te = {}, {}, {}

        for u, i, t in data_tr:
            if u not in seq_tr:
                seq_tr[u] = []
            seq_tr[u].append((i, t))

        for u, i, t in data_va:
            if u not in seq_va:
                seq_va[u] = []
            seq_va[u].append((i, t))

        for u, i, t in data_te:
            if u not in seq_te:
                seq_te[u] = []
            seq_te[u].append(i)

        for u, v in seq_tr.items():
            l = sorted(v, key=lambda x: x[1], reverse=True)
            seq_tr[u] = ','.join([str(p[0]) for p in l])

        for u, v in seq_va.items():
            l = sorted(v, key=lambda x: x[1], reverse=True)
            seq_va[u] = ','.join(str(p[0]) for p in l)

        for u, v in seq_te.items():
            seq_te[u] = ','.join(str(p) for p in seq_te[u])

        format_submit(seq_tr, 'historical_train.csv', submit_dir=DIR)
        format_submit(seq_va, 'res_T.csv', submit_dir=DIR)
        format_submit(seq_te, 'res_T_test.csv', submit_dir=DIR)

        seq_va_tr = seq_va
        for u in seq_tr:
            if u in seq_va:
                seq_va_tr[u] = seq_va[u] + ',' + seq_tr[u]
        format_submit(seq_va_tr, 'historical_train_test.csv', submit_dir=DIR)
        return
