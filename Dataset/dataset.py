import os.path
from tqdm import tqdm
from torch.utils import data
import scipy.io as scio
from utils import *
import pandas as pd
from sklearn.model_selection import KFold

class BasicDataset(data.Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if self.args.dataset == 'Cdataset':
            data = scio.loadmat(os.path.join('Dataset', 'Res', self.args.dataset,  self.args.dataset+'.mat'))
            self.drdi = data['didr'].T
            self.disease_sim = data['disease']
            self.drug_sim = data['drug']
        elif self.args.dataset == 'Gdataset':
            data = scio.loadmat(os.path.join('Dataset', 'Res', self.args.dataset,  self.args.dataset+'.mat'))
            self.drdi = data['didr'].T
            self.disease_sim = data['disease']
            self.drug_sim = data['drug']
        elif self.args.dataset == 'Ldataset':
            self.drdi = np.loadtxt(os.path.join('Dataset', 'Res', self.args.dataset,  'drug_dis.csv'), delimiter=",")
            self.disease_sim =np.loadtxt(os.path.join('Dataset', 'Res', self.args.dataset,  'dis_sim.csv'), delimiter=",")
            self.drug_sim = np.loadtxt(os.path.join('Dataset', 'Res', self.args.dataset,  'drug_sim.csv'), delimiter=",")
        elif self.args.dataset == 'lrssl':
            self.drdi = pd.read_csv(os.path.join('Dataset', 'Res', self.args.dataset,  'drug_dis.txt'), index_col=0, delimiter='\t').values
            self.disease_sim = pd.read_csv(os.path.join('Dataset', 'Res', self.args.dataset,  'dis_sim.txt'), index_col=0, delimiter='\t').values
            self.drug_sim = pd.read_csv(os.path.join('Dataset', 'Res', self.args.dataset,  'drug_sim.txt'), index_col=0, delimiter='\t').values
        else:
            raise ValueError('Invalid dataset')

        self.Wdname_len = self.disease_sim.shape[0]
        self.Wrname_len = self.drug_sim.shape[0]
        self.didr = self.drdi.T

        # for trainloader
        self.cv_data = self.get_cv_data()
        self.sequence = []

    def get_behavior(self, position, id, type):
        if type == 'drug':
            return [x[0] for x in position if x[1] == id]
        elif type == 'disease':
            return [x[1] for x in position if x[0] == id]
        else:
            raise ValueError('Invalid type')

    def get_rating(self, behavior_seq, target_id, type):
        if type == 'drug':
            return [self.drdi[x][target_id] for x in behavior_seq]
        elif type == 'disease':
            return [self.didr[x][target_id] for x in behavior_seq]
        else:
            raise ValueError('Invalid type')

    def get_cv_data(self):
        drug_num, disease_num = self.drdi.shape[0], self.drdi.shape[1]
        drug_id, disease_id = np.nonzero(self.drdi)

        num_len = int(np.ceil(len(drug_id) * 1))  # setting sparse ratio
        drug_id, disease_id = drug_id[0: num_len], disease_id[0: num_len]

        neutral_flag = 0
        labels = np.full((drug_num, disease_num), neutral_flag, dtype=np.int32)
        observed_labels = [1] * len(drug_id)
        labels[drug_id, disease_id] = np.array(observed_labels)

        # negative sampling
        neg_drug_idx, neg_disease_idx = np.where(self.drdi == 0)
        neg_pairs = np.array([[dr, di] for dr, di in zip(neg_drug_idx, neg_disease_idx)])
        np.random.seed(6)
        np.random.shuffle(neg_pairs)

        # positive sampling
        pos_pairs = np.array([[dr, di] for dr, di in zip(drug_id, disease_id)])
        pos_idx = np.array([dr * disease_num + di for dr, di in pos_pairs])

        cv_data = {}
        count = 0
        kfold = KFold(n_splits=10, shuffle=True, random_state=self.args.seed)
        for train_data, test_data in kfold.split(pos_idx):

            pairs_pos_train = pos_pairs[np.array(train_data)]
            pairs_neg_train = neg_pairs[0:len(pairs_pos_train)]
            pairs_train = np.concatenate([random.sample(list(pairs_pos_train), int(len(pairs_pos_train)*self.args.sparse)), pairs_neg_train], axis=0)

            # test dataset contains positive and negative
            idx_pos_test = np.array(pos_idx)[np.array(test_data)]

            pairs_pos_test = pos_pairs[np.array(test_data)]
            pairs_neg_test = neg_pairs[len(pairs_pos_train): len(pairs_pos_train) + len(idx_pos_test) + 1]
            pairs_test = np.concatenate([pairs_pos_test, pairs_neg_test], axis=0)
            cv_data[count] = [pairs_train, pairs_test]
            count += 1

        return cv_data

    def align_seqlen(self):
        original_len = len(self.sequence)
        for i in tqdm(range(original_len), desc="Preparing behavior data"):
            item = self.sequence[i]
            drug_id = item[0]
            disease_id = item[2]
            drug_behavior = item[1]
            disease_behavior = item[3]
            label = item[4]
            if len(drug_behavior) < self.args.seqlen_drug_behavior or len(disease_behavior) < self.args.seqlen_disease_behavior:
                continue

            sub_drug_behaviors = segment_sequence(drug_behavior, self.args.seqlen_drug_behavior, drug_id)
            sub_disease_behaviors = segment_sequence(disease_behavior, self.args.seqlen_disease_behavior, disease_id)
            for drug_behavior in sub_drug_behaviors:
                for disease_behavior in sub_disease_behaviors:
                    drug_rating = torch.tensor(self.get_rating(drug_behavior, disease_id, 'drug'))
                    disease_rating = torch.tensor(self.get_rating(disease_behavior, drug_id, 'disease'))

                    self.sequence.append((drug_id, drug_behavior, drug_rating, disease_id, disease_behavior, disease_rating, label))

        self.sequence = self.sequence[original_len:]
        random.shuffle(self.sequence)

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        return self.sequence[item][0], self.sequence[item][1], self.sequence[item][2], self.sequence[item][3], self.sequence[item][4], self.sequence[item][5], self.sequence[item][6]

class trainDataset(BasicDataset):
    def get_all_sequence_train(self, train_position):
        random.shuffle(train_position)
        for sample in train_position:
            drug_id = sample[0]
            disease_id = sample[1]
            drug_behavior = self.get_behavior(train_position, disease_id, 'drug')
            drug_behavior.remove(drug_id)
            disease_behavior = self.get_behavior(train_position, drug_id, 'disease')
            disease_behavior.remove(disease_id)
            label = self.drdi[drug_id][disease_id]
            self.sequence.append((drug_id, drug_behavior, disease_id, disease_behavior, label))
        self.align_seqlen()

class testDataset(BasicDataset):
    def get_all_sequence_test(self, train_position, test_position):
        for sample in test_position:
            drug_id = sample[0]
            disease_id = sample[1]
            drug_behavior = self.get_behavior(train_position, disease_id, 'drug')
            disease_behavior = self.get_behavior(train_position, drug_id, 'disease')
            label = self.drdi[drug_id][disease_id]
            self.sequence.append((drug_id, drug_behavior, disease_id, disease_behavior, label))
        self.align_seqlen()

class stage1_drug(BasicDataset):
    def __init__(self, args):
        super().__init__(args)

        for i in range(self.Wrname_len):
            for j in range(i, self.Wrname_len):
                self.sequence.append((self.drug_sim[i], self.drug_sim[j], self.drug_sim[i][j]))

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        return self.sequence[item][0], self.sequence[item][1], self.sequence[item][2]

class stage1_disease(BasicDataset):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        for i in range(self.Wdname_len):
            for j in range(i, self.Wdname_len):
                self.sequence.append((self.disease_sim[i], self.disease_sim[j], self.disease_sim[i][j]))

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, item):
        return self.sequence[item][0], self.sequence[item][1], self.sequence[item][2]

class Basic_denova(BasicDataset):
    def __init__(self, drug_id, args):
        super().__init__(args)
        self.drug_id = drug_id

    def get_train_position(self):
        denova_matrix = self.drdi.copy()
        denova_matrix[self.drug_id, :] = 2
        train_position_1 = np.argwhere(denova_matrix == 1)
        train_position_0 = np.argwhere(denova_matrix == 0)
        balance_train_position_1 = train_position_1.tolist()
        balance_train_position_0 = random.sample(list(train_position_0), len(train_position_1))
        train_position = balance_train_position_1 + balance_train_position_0
        return train_position

    def align_seqlen(self):
        original_len = len(self.sequence)
        for i in tqdm(range(original_len), desc="Preparing behavior data"):
            item = self.sequence[i]
            drug_id = item[0]
            disease_id = item[2]
            drug_behavior = item[1]
            label = item[3]
            if len(drug_behavior) < self.args.seqlen_drug_behavior:
                continue

            sub_drug_behaviors = segment_sequence(drug_behavior, self.args.seqlen_drug_behavior, drug_id)
            for drug_behavior in sub_drug_behaviors:
                drug_rating = torch.tensor(self.get_rating(drug_behavior, disease_id, 'drug'))
                self.sequence.append((drug_id, drug_behavior, drug_rating, disease_id, label))

        self.sequence = self.sequence[original_len:]
        random.shuffle(self.sequence)

    def __getitem__(self, item):
        return self.sequence[item][0], self.sequence[item][1], self.sequence[item][2], self.sequence[item][3], self.sequence[item][4]

class denova_drug_train(Basic_denova):
    def __init__(self, drug_id, args):
        super().__init__(drug_id, args)

    def get_all_sequence(self):
        train_position = self.get_train_position()
        random.shuffle(train_position)
        for sample in train_position:
            drug_id = sample[0]
            disease_id = sample[1]
            drug_behavior = self.get_behavior(train_position, disease_id, 'drug')
            drug_behavior.remove(drug_id)
            label = self.drdi[drug_id][disease_id]
            self.sequence.append((drug_id, drug_behavior, disease_id, label))
        self.align_seqlen()

    def __len__(self):
        return len(self.sequence)

class denova_drug_test(Basic_denova):
    def __init__(self, drug_id, args):
        super().__init__(drug_id, args)

    def get_all_sequence(self):
        train_position = self.get_train_position()
        test_position = [[self.drug_id, i] for i in range(self.drdi.shape[1])]
        for sample in test_position:
            drug_id = sample[0]
            disease_id = sample[1]
            drug_behavior = self.get_behavior(train_position, disease_id, 'drug')
            label = self.drdi[drug_id][disease_id]
            self.sequence.append((drug_id, drug_behavior, disease_id, label))
        self.align_seqlen()

    def __len__(self):
        return len(self.sequence)














