import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from collections import defaultdict
from transformers import PreTrainedTokenizer, AutoTokenizer
from utils import create_graph, Collator, dwie_create_graph
from torch.utils.data import DataLoader
from utils import gen_dataset_coref
import spacy


class DocRED(Dataset):
    def __init__(self, data_module, dataset_dir: str, file_name: str, tokenizer: PreTrainedTokenizer,
                 force_regeneration: bool = False, use_coref: bool = True):
        super(DocRED, self).__init__()
        self.data_module = data_module
        self.name = "re-docred"
        dataset_dir = Path(dataset_dir)
        save_dir = dataset_dir / "bin"
        meta_dir = dataset_dir / "meta"
        kg_dir = dataset_dir / "kg"
        with open(meta_dir / "rel2id.json", "r", encoding="utf-8") as f:
            self.rel2id: Dict[str, int] = json.load(f)
        with open(meta_dir / "ner2id.json", "r", encoding="utf-8") as f:
            self.ner2id: Dict[str, int] = json.load(f)
        self.id2rel = {value: key for key, value in self.rel2id.items()}
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_name_or_path = tokenizer.name_or_path
        model_name_or_path = model_name_or_path[model_name_or_path.rfind("/") + 1:]
        if use_coref:
            ori_path = gen_dataset_coref(self.data_module.coref_nlp, dataset_dir, file_name, force_regeneration)
        else:
            ori_path = str(dataset_dir / file_name)
        with open(ori_path, "r") as fh:
            self.data: List[Dict] = json.load(fh)
        split = ori_path[ori_path.rfind("/") + 1:ori_path.rfind(".")]
        kg_path = kg_dir / (file_name[:file_name.rfind('.')] + "_graph.json")
        with open(kg_path, "r") as fh:
            self.kg: List[List[Dict]] = json.load(fh)
        save_path = save_dir / (split + f".{model_name_or_path}.pt")
        if os.path.exists(save_path) and not force_regeneration:
            print(f"Loading DocRED {split} features ...")
            self.features = torch.load(save_path)
        else:
            self.features = self.read_docred(split, tokenizer)
            torch.save(self.features, save_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def read_docred(self, split, tokenizer):
        i_line = 0
        pos_samples = 0
        neg_samples = 0
        features = []

        max_tokens_len = 0

        for docid, doc in tqdm(enumerate(self.data), desc=f"Reading DocRED {split} data", total=len(self.data), ncols=100):
            title: str = doc['title']
            entities: List[List[Dict]] = doc['vertexSet']
            sentences: List[List[str]] = doc['sents']
            ori_labels: List[Dict] = doc.get('labels', [])

            ENT_NUM = len(entities)
            SENT_NUM = len(sentences)
            MEN_NUM = len([m for e in entities for m in e if "coref" not in m])
            COREF_NUM = len([m for e in entities for m in e if "coref" in m])

            entity_start, entity_end = [], []
            for entity in entities:
                for mention in entity:
                    if "coref" not in mention:
                        sent_id: int = mention["sent_id"]
                        pos: List[int] = mention["pos"]
                        entity_start.append((sent_id, pos[0]))
                        entity_end.append((sent_id, pos[1] - 1))
            assert len(entity_start) == len(entity_end) == MEN_NUM
            entity_start = set(entity_start)
            entity_end = set(entity_end)

            tokens: List[str] = []
            word2token: List[List[int]] = []
            for i_s, sent in enumerate(sentences):
                idx_map = [0] * len(sent)
                for i_w, word in enumerate(sent):
                    idx_map[i_w] = len(tokens)
                    word_tokens = tokenizer.tokenize(word)
                    if (i_s, i_w) in entity_start:
                        word_tokens = ["*"] + word_tokens
                    if (i_s, i_w) in entity_end:
                        word_tokens = word_tokens + ["*"]
                    tokens.extend(word_tokens)
                idx_map.append(len(tokens))
                word2token.append(idx_map)

            sent_pos = [(word2token[i][0], word2token[i][-1]) for i in range(SENT_NUM)]

            train_triple = defaultdict(list)
            for label in ori_labels:
                h, t, r, evi = label['h'], label['t'], self.rel2id[label['r']], label['evidence']
                train_triple[h, t].append({'relation': r, "evidence": evi})

            coref_pos = [[] for _ in range(ENT_NUM)]
            entity_pos = [[] for _ in range(ENT_NUM)]
            mention_pos: Tuple[List[int], List[int]] = ([], [])

            entity_types: List[int] = []

            ent2mention: List[List[int]] = [[] for _ in range(ENT_NUM)]
            mention2ent: List[int] = []

            sent2mention: List[List[int]] = [[] for _ in range(SENT_NUM)]
            mention2sent: List[int] = []

            mention_id = 0
            for entity_id, entity in enumerate(entities):
                name_lens = np.array([len(m['name']) for m in entity if "coref" not in m])
                long_idx = np.argmax(name_lens)
                entity_types.append(self.ner2id[entity[long_idx]['type']])
                for mention in entity:
                    sent_id, pos = mention["sent_id"], mention["pos"]
                    start = word2token[sent_id][pos[0]]
                    end = word2token[sent_id][pos[1]]

                    if "coref" in mention:
                        coref_pos[entity_id].append((start, end))
                        continue

                    entity_pos[entity_id].append((start, end))
                    mention_pos[0].append(start)
                    mention_pos[1].append(end)

                    ent2mention[entity_id].append(mention_id)
                    mention2ent.append(entity_id)

                    sent2mention[sent_id].append(mention_id)
                    mention2sent.append(sent_id)

                    mention_id += 1
            assert sum(len(x) for x in coref_pos) == COREF_NUM

            hts: List[List[int]] = []
            relations: List[Tensor] = []
            for h in range(ENT_NUM):
                for t in range(ENT_NUM):
                    hts.append([h, t])
                    relation = torch.zeros(len(self.rel2id))
                    if (h, t) in train_triple:
                        for label in train_triple[h, t]:
                            r, e = label["relation"], label["evidence"]
                            relation[r] = 1.
                        relations.append(relation)
                        pos_samples += 1
                    else:
                        relation[0] = 1.
                        relations.append(relation)
                        neg_samples += 1
            assert len(relations) == len(hts) == ENT_NUM * ENT_NUM
            relations: Tensor = torch.stack(relations)

            max_tokens_len = max(max_tokens_len, len(tokens) + 2)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            assert len(input_ids) == len(tokens) + 2

            men_graph, ent_graph, etype, e_label = create_graph(mention2ent, ent2mention, sent2mention, mention2sent,
                                                                self.kg[docid], self.rel2id, MEN_NUM, ENT_NUM, SENT_NUM,
                                                                1, relations.reshape(ENT_NUM, ENT_NUM, -1))

            i_line += 1
            feature = {
                'title': title,
                'input_ids': input_ids,
                'hts': hts,
                'sent_pos': sent_pos,
                'entity_pos': entity_pos,
                'coref_pos': coref_pos,
                'mention_pos': mention_pos[0],
                'entity_types': entity_types,
                'men_graph': men_graph,
                'ent_graph': ent_graph,
                'etype': etype,
                'e_label': e_label,
                'label': relations,
            }
            features.append(feature)

        print("# of documents {}.".format(i_line))
        print("maximum tokens length:", max_tokens_len)
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))
        return features

    def to_official(self, preds):
        h_idx, t_idx, title = [], [], []

        for f in self.features:
            hts = f["hts"]
            h_idx += [ht[0] for ht in hts]
            t_idx += [ht[1] for ht in hts]
            title += [f["title"] for ht in hts]

        res = []
        for i in range(preds.shape[0]):
            pred = preds[i]
            pred = np.nonzero(pred)[0].tolist()
            for p in pred:
                if p != 0 and p < 97:
                    res.append({
                            'title': title[i],
                            'h_idx': h_idx[i],
                            't_idx': t_idx[i],
                            'r': self.id2rel[p],
                    })
        return res

    def official_evaluate_benchmark(self, preds):
        """
            Adapted from the official evaluation code
        """
        result = self.to_official(preds)
        freq_keys = {'P17', 'P131', 'P27', 'P150', 'P175', 'P577', 'P463', 'P527', 'P495', 'P361'}
        long_tail_keys = set(self.rel2id.keys()) - freq_keys

        fact_in_train_annotated, fact_in_train_distant = self.data_module.gen_train_facts()
        truth = self.data

        std = {}
        std_freq = {}
        std_long_tail = {}
        tot_evidences = 1
        titleset = set([])

        title2vectexSet = {}
        std_intra = {}
        std_inter = {}
        std_inter_long = {}

        def findSmallestDifference(A, B, m, n):

            # Sort both arrays
            # using sort function
            A.sort()
            B.sort()

            a = 0
            b = 0

            # Initialize result as max value
            result = sys.maxsize

            # Scan Both Arrays upto
            # sizeof of the Arrays
            while (a < m and b < n):

                if (abs(A[a] - B[b]) < result):
                    result = abs(A[a] - B[b])

                # Move Smaller Value
                if (A[a] < B[b]):
                    a += 1

                else:
                    b += 1
            # return final sma result
            return result

        for x in truth:
            title = x['title']
            titleset.add(title)

            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet

            for label in x['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                h_sent_set = [x['sent_id'] for x in vertexSet[h_idx]]
                t_sent_set = [x['sent_id'] for x in vertexSet[t_idx]]

                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])
                if findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set)) == 0:
                    std_intra[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if 1 <= findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set)):
                    std_inter[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if 5 < findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set)):
                    std_inter_long[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if r in freq_keys:
                    std_freq[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if r in long_tail_keys:
                    std_long_tail[(title, r, h_idx, t_idx)] = set(label['evidence'])

        tot_relations = len(std)
        tot_relations_freq = len(std_freq)
        tot_relations_long_tail = len(std_long_tail)
        tot_relations_intra = len(std_intra)
        tot_relations_inter = len(std_inter)
        tot_relations_inter_long = len(std_inter_long)

        result.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        if len(result) > 1:
            submission_answer = [result[0]]
            for i in range(1, len(result)):
                x = result[i]
                y = result[i - 1]
                if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                    submission_answer.append(result[i])
        else:
            submission_answer = []
        submission_answer_freq = []
        submission_answer_long_tail = []

        submission_answer_freq = [x for x in submission_answer if x['r'] in freq_keys]
        submission_answer_long_tail = [x for x in submission_answer if x['r'] in long_tail_keys]
        submission_answer_intra = []
        submission_answer_inter = []
        submission_answer_inter_long = []
        for i in range(len(submission_answer)):
            vertexSet = title2vectexSet[submission_answer[i]['title']]
            if title not in title2vectexSet:
                print(title)
                continue
            h_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['h_idx']]]
            t_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['t_idx']]]
            if findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set)) == 0:
                submission_answer_intra.append(submission_answer[i])
            if 1 <= findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set)):
                submission_answer_inter.append(submission_answer[i])
            if 5 < findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set)):
                submission_answer_inter_long.append(submission_answer[i])

        correct_re = 0
        correct_re_freq = 0
        correct_re_long_tail = 0
        correct_re_intra = 0
        correct_re_inter = 0
        correct_re_inter_long = 0
        correct_evidence = 0
        pred_evi = 0

        correct_in_train_annotated = 0
        correct_in_train_distant = 0
        titleset2 = set([])
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if 'evidence' in x:
                evi = set(x['evidence'])
            else:
                evi = set([])
            pred_evi += len(evi)

            if (title, r, h_idx, t_idx) in std:
                correct_re += 1
                stdevi = std[(title, r, h_idx, t_idx)]
                correct_evidence += len(stdevi & evi)
                in_train_annotated = in_train_distant = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True
                        if (n1['name'], n2['name'], r) in fact_in_train_distant:
                            in_train_distant = True

                if in_train_annotated:
                    correct_in_train_annotated += 1
                if in_train_distant:
                    correct_in_train_distant += 1
        for x in submission_answer_freq:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_freq:
                correct_re_freq += 1
        for x in submission_answer_long_tail:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_long_tail:
                correct_re_long_tail += 1

        for x in submission_answer_intra:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_intra:
                correct_re_intra += 1
        for x in submission_answer_inter:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_inter:
                correct_re_inter += 1

        for x in submission_answer_inter_long:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_inter_long:
                correct_re_inter_long += 1

        if len(submission_answer) > 0:
            re_p = 1.0 * correct_re / len(submission_answer)
        else:
            re_p = 0
        re_r = 1.0 * correct_re / tot_relations
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        if len(submission_answer_freq) > 0:
            re_p_freq = 1.0 * correct_re_freq / len(submission_answer_freq)
        else:
            re_p_freq = 0

        re_r_freq = 1.0 * correct_re_freq / tot_relations_freq
        if re_p_freq + re_r_freq == 0:
            re_f1_freq = 0
        else:
            re_f1_freq = 2.0 * re_p_freq * re_r_freq / (re_p_freq + re_r_freq)
        if len(submission_answer_long_tail) > 0:
            re_p_long_tail = 1.0 * correct_re_long_tail / len(submission_answer_long_tail)
        else:
            re_p_long_tail = 0

        re_r_long_tail = 1.0 * correct_re_long_tail / tot_relations_long_tail
        if re_p_long_tail + re_r_long_tail == 0:
            re_f1_long_tail = 0
        else:
            re_f1_long_tail = 2.0 * re_p_long_tail * re_r_long_tail / (re_p_long_tail + re_r_long_tail)

        if len(submission_answer_intra) > 0:
            re_p_intra = 1.0 * correct_re_intra / len(submission_answer_intra)
        else:
            re_p_intra = 0

        re_r_intra = 1.0 * correct_re_intra / tot_relations_intra
        if re_p_intra + re_r_intra == 0:
            re_f1_intra = 0
        else:
            re_f1_intra = 2.0 * re_p_intra * re_r_intra / (re_p_intra + re_r_intra)

        if len(submission_answer_inter) > 0:
            re_p_inter = 1.0 * correct_re_inter / len(submission_answer_inter)
        else:
            re_p_inter = 0
        re_r_inter = 1.0 * correct_re_inter / tot_relations_inter
        if re_p_inter + re_r_inter == 0:
            re_f1_inter = 0
        else:
            re_f1_inter = 2.0 * re_p_inter * re_r_inter / (re_p_inter + re_r_inter)

        evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
        evi_r = 1.0 * correct_evidence / tot_evidences
        if evi_p + evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
                    len(submission_answer) - correct_in_train_annotated + 1e-5)
        re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (
                    len(submission_answer) - correct_in_train_distant + 1e-5)

        if re_p_ignore_train_annotated + re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (
                        re_p_ignore_train_annotated + re_r)

        if re_p_ignore_train + re_r == 0:
            re_f1_ignore_train = 0
        else:
            re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

        return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train, re_p, re_r, re_f1_freq, re_f1_long_tail, re_f1_intra, re_f1_inter, re_p_freq, re_r_freq, re_p_long_tail, re_r_long_tail


class DWIE(Dataset):
    def __init__(self, data_module, dataset_dir: str, file_name: str, tokenizer: PreTrainedTokenizer,
                       force_regeneration: bool = False, use_coref: bool = True, max_seq_length: int = 1024):
        super(DWIE, self).__init__()
        self.data_module = data_module
        self.name = 'dwie'
        dataset_dir = Path(dataset_dir)
        save_dir = dataset_dir / "bin"
        meta_dir = dataset_dir / "meta"
        kg_dir = dataset_dir / "kg"
        with open(meta_dir / "rel2id.json", "r", encoding="utf-8") as f:
            self.rel2id: Dict[str, int] = json.load(f)
        with open(meta_dir / "kg_rel2id.json", "r", encoding="utf-8") as f:
            self.kg_rel2id: Dict[str, int] = json.load(f)
        with open(meta_dir / "ner2id.json", "r", encoding="utf-8") as f:
            self.ner2id: Dict[str, int] = json.load(f)
        self.id2rel = {value: key for key, value in self.rel2id.items()}
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        model_name_or_path = tokenizer.name_or_path
        model_name_or_path = model_name_or_path[model_name_or_path.rfind("/") + 1:]
        if use_coref:
            ori_path = gen_dataset_coref(self.data_module.coref_nlp, dataset_dir, file_name, force_regeneration)
        else:
            ori_path = str(dataset_dir / file_name)
        with open(ori_path, "r", encoding='utf-8') as fh:
            self.data: List[Dict] = json.load(fh)
        split = ori_path[ori_path.rfind("/") + 1:ori_path.rfind(".")]
        kg_path = kg_dir / (file_name[:file_name.rfind('.')] + "_graph.json")
        with open(kg_path, "r") as fh:
            self.kg: List[List[Dict]] = json.load(fh)
        save_path = save_dir / (split + f".{model_name_or_path}.pt")
        if os.path.exists(save_path) and not force_regeneration:
            print(f"Loading DWIE {split} features ...")
            self.features = torch.load(save_path)
        else:
            self.features = self.read_dwie(split, tokenizer, max_seq_length)
            torch.save(self.features, save_path)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def read_dwie(self, split, tokenizer, max_seq_length):
        i_line = 0
        pos_samples = 0
        neg_samples = 0
        features = []

        max_tokens_len = 0
        lq512 = 0
        lq1024 = 0
        lq1536 = 0
        lq2048 = 0
        lq2560 = 0

        for docid, doc in tqdm(enumerate(self.data), desc=f"Reading DWIE {split} data", total=len(self.data), ncols=100):
            title: str = str(doc['id'])
            entities: List[List[Dict]] = doc['vertexSet']
            sentences: List[List[str]] = doc['sents']
            ori_labels: List[Dict] = doc.get('labels', [])

            ENT_NUM = len(entities)
            SENT_NUM = len(sentences)
            MEN_NUM = len([m for e in entities for m in e if "coref" not in m])
            COREF_NUM = len([m for e in entities for m in e if "coref" in m])

            entity_start, entity_end = [], []
            for entity in entities:
                for mention in entity:
                    if "coref" not in mention:
                        sent_id: int = mention["sent_id"]
                        pos: List[int] = mention["pos"]
                        entity_start.append((sent_id, pos[0]))
                        entity_end.append((sent_id, pos[1] - 1))
            assert len(entity_start) == len(entity_end) == MEN_NUM
            entity_start = set(entity_start)
            entity_end = set(entity_end)

            tokens: List[str] = []
            word2token: List[List[int]] = []
            for i_s, sent in enumerate(sentences):
                idx_map = [0] * len(sent)
                for i_w, word in enumerate(sent):
                    idx_map[i_w] = len(tokens)
                    word_tokens = tokenizer.tokenize(word)
                    if (i_s, i_w) in entity_start:
                        word_tokens = ["*"] + word_tokens
                    if (i_s, i_w) in entity_end:
                        word_tokens = word_tokens + ["*"]
                    tokens.extend(word_tokens)
                idx_map.append(len(tokens))
                word2token.append(idx_map)

            sent_pos = [(word2token[i][0], word2token[i][-1]) for i in range(SENT_NUM)]

            train_triple = defaultdict(list)
            for label in ori_labels:
                h, t, r = label['h'], label['t'], self.rel2id[label['r']]
                train_triple[h, t].append({'relation': r})

            coref_pos = [[] for _ in range(ENT_NUM)]
            entity_pos = [[] for _ in range(ENT_NUM)]
            mention_pos: Tuple[List[int], List[int]] = ([], [])

            entity_types: List[int] = []

            ent2mention: List[List[int]] = [[] for _ in range(ENT_NUM)]
            mention2ent: List[int] = []

            sent2mention: List[List[int]] = [[] for _ in range(SENT_NUM)]
            mention2sent: List[int] = []

            mention_id = 0
            for entity_id, entity in enumerate(entities):
                name_lens = np.array([len(m['name']) for m in entity if "coref" not in m])
                long_idx = np.argmax(name_lens)
                entity_types.append(self.ner2id[entity[long_idx]['type']])
                for mention in entity:
                    sent_id, pos = mention["sent_id"], mention["pos"]
                    start = word2token[sent_id][pos[0]]
                    end = word2token[sent_id][pos[1]]

                    if "coref" in mention:
                        coref_pos[entity_id].append((start, end))
                        continue

                    entity_pos[entity_id].append((start, end))
                    mention_pos[0].append(start)
                    mention_pos[1].append(end)

                    ent2mention[entity_id].append(mention_id)
                    mention2ent.append(entity_id)

                    sent2mention[sent_id].append(mention_id)
                    mention2sent.append(sent_id)

                    mention_id += 1
            assert sum(len(x) for x in coref_pos) == COREF_NUM

            hts: List[List[int]] = []
            relations: List[Tensor] = []
            for h in range(ENT_NUM):
                for t in range(ENT_NUM):
                    hts.append([h, t])
                    relation = torch.zeros(len(self.rel2id))
                    if (h, t) in train_triple:
                        for label in train_triple[h, t]:
                            r = label["relation"]
                            relation[r] = 1.
                        relations.append(relation)
                        pos_samples += 1
                    else:
                        relation[0] = 1.
                        relations.append(relation)
                        neg_samples += 1
            assert len(relations) == len(hts) == ENT_NUM * ENT_NUM
            relations: Tensor = torch.stack(relations)
            # if len(tokens) + 2 > 2048:
            #     print(f"# {docid} doc has {len(tokens) + 2} tokens > 2048 ")

            max_tokens_len = max(max_tokens_len, len(tokens) + 2)
            x = len(tokens) + 2
            if x <= 512:
                lq512 += 1
            if x <= 1024:
                lq1024 += 1
            if x <= 1536:
                lq1536 += 1
            if x <= 2048:
                lq2048 += 1
            if x <= 2560:
                lq2560 += 1

            tokens = tokens[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            assert len(input_ids) == len(tokens) + 2

            men_graph, ent_graph, etype, e_label = dwie_create_graph(
                                                                    mention2ent, ent2mention, sent2mention, mention2sent,
                                                                    self.kg[docid], self.kg_rel2id, MEN_NUM, ENT_NUM, SENT_NUM,
                                                                    1, relations.reshape(ENT_NUM, ENT_NUM, -1))

            i_line += 1
            feature = {
                'title': title,
                'input_ids': input_ids,
                'hts': hts,
                'sent_pos': sent_pos,
                'entity_pos': entity_pos,
                'coref_pos': coref_pos,
                'mention_pos': mention_pos[0],
                'entity_types': entity_types,
                'men_graph': men_graph,
                'ent_graph': ent_graph,
                'etype': etype,
                'e_label': e_label,
                'label': relations,
            }
            features.append(feature)

        print("# of documents {}.".format(i_line))
        print("maximum tokens length:", max_tokens_len)
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))
        print("<=512: ", lq512)
        print("<=1024: ", lq1024)
        print("<=1536: ", lq1536)
        print("<=2048: ", lq2048)
        print("<=2560: ", lq2560)
        return features

    def to_official(self, preds):
        h_idx, t_idx, title = [], [], []

        for f in self.features:
            hts = f["hts"]
            h_idx += [ht[0] for ht in hts]
            t_idx += [ht[1] for ht in hts]
            title += [f["title"] for ht in hts]

        res = []
        for i in range(preds.shape[0]):
            pred = preds[i]
            pred = np.nonzero(pred)[0].tolist()
            for p in pred:
                if p != 0 and p < 66:
                    res.append({
                            'title': title[i],
                            'h_idx': h_idx[i],
                            't_idx': t_idx[i],
                            'r': self.id2rel[p],
                    })
        return res

    def official_evaluate(self, preds):
        """
            Adapted from the official evaluation code
        """
        tmp = self.to_official(preds)
        truth = self.data
        fact_in_train_annotated = self.data_module.gen_train_facts()

        std = {}
        tot_evidences = 0
        titleset = set([])

        title2vectexSet = {}

        for x in truth:
            title = x['id']
            titleset.add(title)

            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet

            for label in x['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                label['evidence'] = []
                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])

        tot_relations = len(std)
        tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        if not tmp:
            submission_answer = []
        else:
            submission_answer = [tmp[0]]
            for i in range(1, len(tmp)):
                x = tmp[i]
                y = tmp[i - 1]
                if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                    submission_answer.append(tmp[i])

        correct_re = 0
        correct_evidence = 0
        pred_evi = 0

        correct_in_train_annotated = 0
        titleset2 = set([])
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if 'evidence' in x:
                evi = set(x['evidence'])
            else:
                evi = set([])
            pred_evi += len(evi)

            if (title, r, h_idx, t_idx) in std:
                correct_re += 1
                stdevi = std[(title, r, h_idx, t_idx)]
                correct_evidence += len(stdevi & evi)
                in_train_annotated = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True

                if in_train_annotated:
                    correct_in_train_annotated += 1

        re_p = 1.0 * correct_re / len(submission_answer) if len(submission_answer) > 0 else 0
        re_r = 1.0 * correct_re / tot_relations
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
        evi_r = 1.0 * correct_evidence / tot_evidences if tot_evidences != 0 else 0
        if evi_p + evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
                    len(submission_answer) - correct_in_train_annotated + 1e-5)

        if re_p_ignore_train_annotated + re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (
                        re_p_ignore_train_annotated + re_r)

        return re_f1_ignore_train_annotated, re_f1, re_p, re_r

    def official_evaluate_benchmark(self, preds):
        result = self.to_official(preds)
        freq_keys = {'in0-x', 'agent_of', 'citizen_of', 'based_in0', 'gpe0', 'in0', 'member_of', 'citizen_of-x', 'head_of', 'based_in0-x'}
        long_tail_keys = set(self.rel2id.keys()) - freq_keys

        fact_in_train_annotated = self.data_module.gen_train_facts()
        truth = self.data

        std = {}
        std_freq = {}
        std_long_tail = {}
        tot_evidences = 1
        titleset = set([])

        title2vectexSet = {}
        std_intra = {}
        std_inter = {}
        std_inter_1 = {}
        std_inter_2 = {}
        std_inter_3 = {}
        std_inter_4 = {}
        std_inter_gt4 = {}
        std_inter_long = {}

        def findSmallestDifference(A, B, m, n):

            # Sort both arrays
            # using sort function
            A.sort()
            B.sort()

            a = 0
            b = 0

            # Initialize result as max value
            result = sys.maxsize

            # Scan Both Arrays upto
            # sizeof of the Arrays
            while (a < m and b < n):

                if (abs(A[a] - B[b]) < result):
                    result = abs(A[a] - B[b])

                # Move Smaller Value
                if (A[a] < B[b]):
                    a += 1

                else:
                    b += 1
            # return final sma result
            return result

        for x in truth:
            title = x['title']
            titleset.add(title)

            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet

            for label in x['labels']:
                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                h_sent_set = [x['sent_id'] for x in vertexSet[h_idx]]
                t_sent_set = [x['sent_id'] for x in vertexSet[t_idx]]

                std[(title, r, h_idx, t_idx)] = set(label['evidence'])
                tot_evidences += len(label['evidence'])
                smallestDistance = findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set))
                if smallestDistance == 0:  # 
                    std_intra[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if smallestDistance >= 1:  # 
                    std_inter[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if smallestDistance == 1:  # 
                    std_inter_1[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if smallestDistance == 2:  # 
                    std_inter_2[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if smallestDistance == 3:  # 
                    std_inter_3[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if smallestDistance == 4:  # 
                    std_inter_4[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if smallestDistance >= 4:  # 
                    std_inter_gt4[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if smallestDistance >= 5:
                    std_inter_long[(title, r, h_idx, t_idx)] = set(label['evidence'])

                if r in freq_keys:
                    std_freq[(title, r, h_idx, t_idx)] = set(label['evidence'])
                if r in long_tail_keys:
                    std_long_tail[(title, r, h_idx, t_idx)] = set(label['evidence'])

        tot_relations = len(std)
        tot_relations_freq = len(std_freq)
        tot_relations_long_tail = len(std_long_tail)
        tot_relations_intra = len(std_intra)
        tot_relations_inter = len(std_inter)
        tot_relations_inter_1 = len(std_inter_1)
        tot_relations_inter_2 = len(std_inter_2)
        tot_relations_inter_3 = len(std_inter_3)
        tot_relations_inter_4 = len(std_inter_4)
        tot_relations_inter_gt4 = len(std_inter_gt4)
        tot_relations_inter_long = len(std_inter_long)

        result.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        if len(result) > 1:
            submission_answer = [result[0]]
            for i in range(1, len(result)):
                x = result[i]
                y = result[i - 1]
                if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                    submission_answer.append(result[i])
        else:
            submission_answer = []
        submission_answer_freq = []
        submission_answer_long_tail = []

        submission_answer_freq = [x for x in submission_answer if x['r'] in freq_keys]
        submission_answer_long_tail = [x for x in submission_answer if x['r'] in long_tail_keys]
        submission_answer_intra = []
        submission_answer_inter = []
        submission_answer_inter_1 = []
        submission_answer_inter_2 = []
        submission_answer_inter_3 = []
        submission_answer_inter_4 = []
        submission_answer_inter_gt4 = []
        submission_answer_inter_long = []
        for i in range(len(submission_answer)):
            vertexSet = title2vectexSet[submission_answer[i]['title']]
            if title not in title2vectexSet:
                print(title)
                continue
            h_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['h_idx']]]
            t_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['t_idx']]]
            smallestDistance = findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set), len(t_sent_set))
            if smallestDistance == 0:
                submission_answer_intra.append(submission_answer[i])
            if smallestDistance >= 1:
                submission_answer_inter.append(submission_answer[i])
            if smallestDistance == 1:
                submission_answer_inter_1.append(submission_answer[i])
            if smallestDistance == 2:
                submission_answer_inter_2.append(submission_answer[i])
            if smallestDistance == 3:
                submission_answer_inter_3.append(submission_answer[i])
            if smallestDistance == 4:
                submission_answer_inter_4.append(submission_answer[i])
            if smallestDistance >= 4:
                submission_answer_inter_gt4.append(submission_answer[i])
            if smallestDistance >= 5:
                submission_answer_inter_long.append(submission_answer[i])

        correct_re = 0
        correct_re_freq = 0
        correct_re_long_tail = 0
        correct_re_intra = 0
        correct_re_inter = 0
        correct_re_inter_1 = 0
        correct_re_inter_2 = 0
        correct_re_inter_3 = 0
        correct_re_inter_4 = 0
        correct_re_inter_gt4 = 0
        correct_re_inter_long = 0
        correct_evidence = 0
        pred_evi = 0

        correct_in_train_annotated = 0
        correct_in_train_distant = 0
        titleset2 = set([])
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if 'evidence' in x:
                evi = set(x['evidence'])
            else:
                evi = set([])
            pred_evi += len(evi)

            if (title, r, h_idx, t_idx) in std:
                correct_re += 1
                stdevi = std[(title, r, h_idx, t_idx)]
                correct_evidence += len(stdevi & evi)
                in_train_annotated = in_train_distant = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_distant = True

                if in_train_annotated:
                    correct_in_train_annotated += 1
                if in_train_distant:
                    correct_in_train_distant += 1
        for x in submission_answer_freq:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_freq:
                correct_re_freq += 1
        for x in submission_answer_long_tail:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_long_tail:
                correct_re_long_tail += 1

        for x in submission_answer_intra:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_intra:
                correct_re_intra += 1
        for x in submission_answer_inter:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_inter:
                correct_re_inter += 1
        for x in submission_answer_inter_1:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_inter_1:
                correct_re_inter_1 += 1
        for x in submission_answer_inter_2:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_inter_2:
                correct_re_inter_2 += 1
        for x in submission_answer_inter_3:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_inter_3:
                correct_re_inter_3 += 1
        for x in submission_answer_inter_4:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_inter_4:
                correct_re_inter_4 += 1

        for x in submission_answer_inter_gt4:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_inter_gt4:
                correct_re_inter_gt4 += 1

        for x in submission_answer_inter_long:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']
            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if (title, r, h_idx, t_idx) in std_inter_long:
                correct_re_inter_long += 1

        if len(submission_answer) > 0:
            re_p = 1.0 * correct_re / len(submission_answer)
        else:
            re_p = 0
        re_r = 1.0 * correct_re / tot_relations
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        if len(submission_answer_freq) > 0:
            re_p_freq = 1.0 * correct_re_freq / len(submission_answer_freq)
        else:
            re_p_freq = 0

        re_r_freq = 1.0 * correct_re_freq / tot_relations_freq
        if re_p_freq + re_r_freq == 0:
            re_f1_freq = 0
        else:
            re_f1_freq = 2.0 * re_p_freq * re_r_freq / (re_p_freq + re_r_freq)
        if len(submission_answer_long_tail) > 0:
            re_p_long_tail = 1.0 * correct_re_long_tail / len(submission_answer_long_tail)
        else:
            re_p_long_tail = 0

        re_r_long_tail = 1.0 * correct_re_long_tail / tot_relations_long_tail
        if re_p_long_tail + re_r_long_tail == 0:
            re_f1_long_tail = 0
        else:
            re_f1_long_tail = 2.0 * re_p_long_tail * re_r_long_tail / (re_p_long_tail + re_r_long_tail)

        if len(submission_answer_intra) > 0:
            re_p_intra = 1.0 * correct_re_intra / len(submission_answer_intra)
        else:
            re_p_intra = 0

        re_r_intra = 1.0 * correct_re_intra / tot_relations_intra
        if re_p_intra + re_r_intra == 0:
            re_f1_intra = 0
        else:
            re_f1_intra = 2.0 * re_p_intra * re_r_intra / (re_p_intra + re_r_intra)

        if len(submission_answer_inter) > 0:
            re_p_inter = 1.0 * correct_re_inter / len(submission_answer_inter)
        else:
            re_p_inter = 0
        re_r_inter = 1.0 * correct_re_inter / tot_relations_inter
        if re_p_inter + re_r_inter == 0:
            re_f1_inter = 0
        else:
            re_f1_inter = 2.0 * re_p_inter * re_r_inter / (re_p_inter + re_r_inter)

        def cal_f1(sa, correct_num, total_num):
            if len(sa) > 0:
                _p = 1.0 * correct_num / len(sa)
            else:
                _p = 0
            if total_num > 0:
                _r = 1.0 * correct_num / total_num
            else:
                _r = 0
            if _p + _r == 0:
                _f1 = 0
            else:
                _f1 = 2.0 * _p * _r / (_p + _r)
            return _p, _r, _f1
        re_p_inter_1, re_r_inter_1, re_f1_inter_1 = cal_f1(submission_answer_inter_1, correct_re_inter_1, tot_relations_inter_1)
        re_p_inter_2, re_r_inter_2, re_f1_inter_2 = cal_f1(submission_answer_inter_2, correct_re_inter_2, tot_relations_inter_2)
        re_p_inter_3, re_r_inter_3, re_f1_inter_3 = cal_f1(submission_answer_inter_3, correct_re_inter_3, tot_relations_inter_3)
        re_p_inter_4, re_r_inter_4, re_f1_inter_4 = cal_f1(submission_answer_inter_4, correct_re_inter_4, tot_relations_inter_4)
        re_p_inter_gt4, re_r_inter_gt4, re_f1_inter_gt4 = cal_f1(submission_answer_inter_gt4, correct_re_inter_gt4, tot_relations_inter_gt4)
        re_p_inter_long, re_r_inter_long, re_f1_inter_long = cal_f1(submission_answer_inter_long, correct_re_inter_long, tot_relations_inter_long)
        print("==================================================================")
        print("intra_f1: ", re_f1_intra * 100, "%: ", tot_relations_intra / tot_relations)
        print("inter_1_f1: ", re_f1_inter_1 * 100, "%: ", tot_relations_inter_1 / tot_relations)
        print("inter_2_f1: ", re_f1_inter_2 * 100, "%: ", tot_relations_inter_2 / tot_relations)
        print("inter_3_f1: ", re_f1_inter_3 * 100, "%: ", tot_relations_inter_3 / tot_relations)
        print("inter_4_f1: ", re_f1_inter_4 * 100, "%: ", tot_relations_inter_4 / tot_relations)
        print("inter_gt4_f1: ", re_f1_inter_gt4 * 100)
        print("inter_long_f1: ", re_f1_inter_long * 100)
        print("==================================================================")
        
        evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
        evi_r = 1.0 * correct_evidence / tot_evidences
        if evi_p + evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
                    len(submission_answer) - correct_in_train_annotated + 1e-5)
        re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (
                    len(submission_answer) - correct_in_train_distant + 1e-5)

        if re_p_ignore_train_annotated + re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (
                        re_p_ignore_train_annotated + re_r)

        if re_p_ignore_train + re_r == 0:
            re_f1_ignore_train = 0
        else:
            re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

        return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train, re_p, re_r, re_f1_freq, re_f1_long_tail, re_f1_intra, re_f1_inter, re_p_freq, re_r_freq, re_p_long_tail, re_r_long_tail

class DocREDataModule:
    def __init__(
            self,
            dataset_dir: str,
            tokenizer: PreTrainedTokenizer,
            train_file: str,
            train_distant_file: str,
            dev_file: str,
            test_file: str,
            force_regeneration: bool,
            use_coref: bool,
            train_batch_size: int,
            test_batch_size: int
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.train_file = train_file
        self.train_distant_file = train_distant_file

        self.collate_fnt = Collator(tokenizer)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.coref_nlp = spacy.load("en_coreference_web_trf")

        self.data_train = DocRED(self, dataset_dir, train_file, tokenizer, force_regeneration, use_coref)

        self.data_dev = DocRED(self, dataset_dir, dev_file, tokenizer, force_regeneration, use_coref)

        self.data_test = DocRED(self, dataset_dir, test_file, tokenizer, force_regeneration, use_coref)

    def gen_train_facts(self):
        data_file_names = [self.train_file, self.train_distant_file]
        truth_dir = os.path.join(self.dataset_dir, 'ref')
        if not os.path.exists(truth_dir):
            os.makedirs(truth_dir)

        facts = []
        for data_file_name in data_file_names:
            fact_file_name = data_file_name[data_file_name.find("train_"):]
            fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

            if os.path.exists(fact_file_name):
                fact_in_train = set([])
                triples = json.load(open(fact_file_name))
                for x in triples:
                    fact_in_train.add(tuple(x))
                facts.append(fact_in_train)
                continue

            fact_in_train = set([])
            ori_data = json.load(open(os.path.join(self.dataset_dir, data_file_name)))
            for data in ori_data:
                vertexSet = data['vertexSet']
                for label in data['labels']:
                    rel = label['r']
                    for n1 in vertexSet[label['h']]:
                        for n2 in vertexSet[label['t']]:
                            fact_in_train.add((n1['name'], n2['name'], rel))

            json.dump(list(fact_in_train), open(fact_file_name, "w"))

            facts.append(fact_in_train)
        return facts

    @property
    def train_dataset(self):
        return self.data_train

    @property
    def dev_dataset(self):
        return self.data_dev

    @property
    def test_dataset(self):
        return self.data_test

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fnt,
        )

    def dev_dataloader(self):
        return DataLoader(
            dataset=self.data_dev,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fnt,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fnt,
        )


class DWIEDataModule:
    def __init__(
            self,
            dataset_dir: str,
            tokenizer: PreTrainedTokenizer,
            train_file: str,
            dev_file: str,
            test_file: str,
            dev_test_file: str,
            max_seq_length: int,
            force_regeneration: bool,
            use_coref: bool,
            train_batch_size: int,
            test_batch_size: int
    ):
        super().__init__()
        self.collate_fnt = Collator(tokenizer)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_file = train_file
        self.dataset_dir = dataset_dir
        self.max_seq_length = max_seq_length

        self.coref_nlp = spacy.load("en_coreference_web_trf")

        self.data_train = DWIE(self, dataset_dir, train_file, tokenizer, force_regeneration, use_coref, max_seq_length)

        self.data_dev = DWIE(self, dataset_dir, dev_file, tokenizer, force_regeneration, use_coref, max_seq_length)

        self.data_test = DWIE(self, dataset_dir, test_file, tokenizer, force_regeneration, use_coref, max_seq_length)

        self.data_dev_test = DWIE(self, dataset_dir, dev_test_file, tokenizer, force_regeneration, use_coref, max_seq_length)

    def gen_train_facts(self):
        data_file_name = self.train_file
        truth_dir = os.path.join(self.dataset_dir, 'ref')
        if not os.path.exists(truth_dir):
            os.makedirs(truth_dir)

        fact_file_name = os.path.join(truth_dir, data_file_name.replace(".json", ".fact"))

        if os.path.exists(fact_file_name):
            fact_in_train = set([])
            triples = json.load(open(fact_file_name))
            for x in triples:
                fact_in_train.add(tuple(x))
            return fact_in_train

        fact_in_train = set([])
        ori_data = json.load(open(os.path.join(self.dataset_dir, data_file_name)))
        for data in ori_data:
            vertexSet = data['vertexSet']
            for label in data['labels']:
                rel = label['r']
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))

        json.dump(list(fact_in_train), open(fact_file_name, "w"))
        return fact_in_train

    @property
    def train_dataset(self):
        return self.data_train

    @property
    def dev_dataset(self):
        return self.data_dev

    @property
    def test_dataset(self):
        return self.data_test
    
    @property
    def dev_test_dataset(self):
        return self.data_dev_test

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collate_fnt,
        )

    def dev_dataloader(self):
        return DataLoader(
            dataset=self.data_dev,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fnt,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fnt,
        )
    
    def dev_test_dataloader(self):
        return DataLoader(
            dataset=self.data_dev_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fnt,
        )


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('./PLM/longformer-large-4096')
    dm = DWIEDataModule('./data/DWIE', tokenizer, 'train.json', 'dev.json', 'test.json', 4096, False, False, 4, 4)
    dm.gen_train_facts()
    dm.data_train.official_evaluate_benchmark(torch.tensor([]))
