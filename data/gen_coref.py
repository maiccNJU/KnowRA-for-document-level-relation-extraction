import json
from collections import defaultdict, Counter
import spacy
from spacy.tokens import Doc
from tqdm import tqdm
import numpy as np


def gen_coref(coref_nlp, doc_id, sample):
    sents = sample['sents']
    entities = sample['vertexSet']
    document = ''
    word2char = []
    word2sent = []
    sent2word = []
    word_cnt = 0
    for sent_id, sent in enumerate(sents):
        sent2word.append([])
        for word_id, word in enumerate(sent):
            word2char.append([])
            word2sent.append([sent_id, word_id])
            word2char[-1].append(len(document))
            document += word
            word2char[-1].append(len(document))
            document += ' '
            sent2word[-1].append(word_cnt)
            word_cnt += 1
    assert len(word2char) == len(word2sent) == sum(len(sent) for sent in sents) == word_cnt
    document = document[:-1]
    WORD_NUM, CHAR_NUM = len(word2char), len(document)
    char2word = np.array([-1] * CHAR_NUM)
    for word_id, (start_idx, end_idx) in enumerate(word2char):
        char2word[start_idx:end_idx] = word_id
    doc = coref_nlp(document)
    clusters = [val for key, val in doc.spans.items() if key.startswith("coref_cluster")]
    char2cluster = np.array([-1] * CHAR_NUM)
    for cluster_id, cluster in enumerate(clusters):
        for mention_span in cluster:
            span = char2cluster[mention_span.start_char:mention_span.end_char]
            span[span == -1] = cluster_id
    char2entity = np.array([-1] * CHAR_NUM)
    entity_clusters = defaultdict(Counter)
    for entity_id, entity in enumerate(entities):
        for mention_id, mention in enumerate(entity):
            sent_id, start_word, end_word = mention['sent_id'], mention['pos'][0], mention['pos'][1] - 1
            start_word_idx, end_word_idx = sent2word[sent_id][start_word], sent2word[sent_id][end_word]
            start_idx, end_idx = word2char[start_word_idx][0], word2char[end_word_idx][1]
            char2entity[start_idx:end_idx] = entity_id
            cluster_id = set(np.unique(char2cluster[start_idx:end_idx]))
            cluster_id.discard(-1)
            # assert len(cluster_id) <= 1, f"{doc_id}/{entity_id}/{mention_id}"
            if len(cluster_id) > 1 and entity_id in entity_clusters:
                del entity_clusters[entity_id]
                break
            if cluster_id:
                entity_clusters[entity_id][cluster_id.pop()] += 1
    entities = sample['vertexSet']
    for entity_id, entity_cluster in entity_clusters.items():
        # assert len(entity_cluster) == 1, f"{doc_id}/{entity_id}/{entity_cluster}"
        max_time = -1
        cluster_id = -1
        for k, v in entity_cluster.items():
            if v > max_time:
                cluster_id, max_time = k, v
        # cluster_id = entity_cluster.pop()
        cluster = clusters[cluster_id]
        for mention_span in cluster:
            if all(np.unique(char2entity[mention_span.start_char:mention_span.end_char]) == -1):
                word_ids = np.unique(char2word[mention_span.start_char:mention_span.end_char])
                word_ids = sorted(list(word_ids))
                if -1 in word_ids:
                    word_ids.pop(0)
                start_sent_id, start_word_id = word2sent[word_ids[0]]
                end_sent_id, end_word_id = word2sent[word_ids[-1]]
                assert start_sent_id == end_sent_id, f"{doc_id}/{entity_id}/{mention_span.text}"
                entities[entity_id].append({
                    "sent_id": start_sent_id,
                    "pos": [start_word_id, end_word_id + 1],
                    "name": document[mention_span.start_char:mention_span.end_char],
                    "type": entities[entity_id][0]['type'],
                    "coref": True
                })
    return sample


def gen_dataset_coref(coref_nlp, filename, split):
    dataset = json.load(open(filename))
    for doc_id, sample in tqdm(enumerate(dataset), desc=f"gen {split} data coref:", ncols=100, total=len(dataset)):
        gen_coref(coref_nlp, doc_id, sample)
    json.dump(dataset, open(f"{split}_coref.json", "w"))


# Define lightweight function for resolving references in text
def resolve_references(doc: Doc) -> str:
    """Function for resolving references with the coref ouput
    doc (Doc): The Doc object processed by the coref pipeline
    RETURNS (str): The Doc string with resolved references
    """
    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_

            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string


if __name__ == '__main__':
    coref_nlp = spacy.load("en_coreference_web_trf")
    gen_dataset_coref(coref_nlp, "test.json", "test")
