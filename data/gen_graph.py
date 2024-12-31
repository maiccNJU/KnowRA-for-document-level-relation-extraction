import requests
from bs4 import BeautifulSoup
import json
from typing import List, Dict
from tqdm import tqdm
from wikidata.client import Client
from collections import defaultdict
import re
from urllib.error import HTTPError


def get_label(label):
    wikidata_url = 'https://www.wikidata.org/w/index.php?sort=relevance&search=' + label + '&title=Special%3ASearch&profile=advanced&fulltext=1&advancedSearch-current=%7B%7D&ns0=1&ns120=1'
    results = requests.get(url=wikidata_url, headers={"Content-Type": "application/json;charset=UTF-8", })
    soup = BeautifulSoup(results.text, 'html.parser', from_encoding='utf-8')
    results = soup.find_all("div", "mw-search-result-heading")
    return [(result.a['href'], result.a['title']) for result in results]


def gen_dataset(filename, split):
    with open(filename, "r") as fh:
        data = json.load(fh)
    name2label = {}
    for docid, doc in tqdm(enumerate(data), desc=f"Reading DocRED {split} data", total=len(data), ncols=80):
        title: str = doc['title']
        entities: List[List[Dict]] = doc['vertexSet']
        sentences: List[List[str]] = doc['sents']
        ori_labels: List[Dict] = doc.get('labels', [])
        for entity in entities:
            for mention in entity:
                name = mention['name']
                if name in name2label:
                    continue
                labels = get_label(name)
                name2label[name] = labels
    json.dump(name2label, open(f"{split}_labels.json", "w"))


docred_rels = json.load(open("rel2id.json"))
client = Client()


def gen_one_hop_relations(k, label):
    try:
        entity = client.get(label, load=True)
    except:
        print(k, label)
        return {"description": "", "aliases": [], "relations": {}}
    if 'en' in entity.attributes['aliases']:
        aliases = [alias['value'] for alias in entity.attributes['aliases']['en']]
    else:
        aliases = []
    if 'en' in entity.description.texts:
        description = entity.description.texts['en']
    else:
        description = ""
    relations = defaultdict(list)
    for rel, obs in entity.attributes['claims'].items():
        if rel not in docred_rels:
            continue
        for ob in obs:
            if 'datavalue' not in ob['mainsnak']:
                continue
            value = ob['mainsnak']['datavalue']
            if value['type'] != 'wikibase-entityid':
                continue
            # if value['type'] == 'wikibase-entityid':
            relations[rel].append(value['value']['id'])
            # elif value['type'] == 'time':
            #     relations[rel].append(value['value']['time'])
            # elif value['type'] == 'quantity':
            #     relations[rel].append(value['value']['amount'])
            # else:
            #     raise Exception()
    return {"description": description, "aliases": aliases, "relations": relations}


def gen_dataset_one_hop_relations(filename, split):
    labels = json.load(open(filename))
    ans = {}
    for k, v in tqdm(labels.items(), desc=f"gen {split} one hop relation:", ncols=100):
        for _id, description in v[:5]:
            new_id = _id[_id.rfind("/") + 1:]
            if not re.fullmatch(r'Q\d+', new_id):
                continue
            if new_id in ans:
                continue
            rels = gen_one_hop_relations(k, new_id)
            ans[new_id] = rels
    json.dump(ans, open(f"{split}_one_hop.json", "w"))


def gen_doc_graph(entity_labels, dataset_one_hop, true_labels):
    label2id = defaultdict(list)
    for entity_id, labels in enumerate(entity_labels):
        for label in labels:
            # if label in label2id:
            #     raise Exception("different entity has same label")
            label2id[label].append(entity_id)
    graph = [defaultdict(list) for _ in range(len(entity_labels))]
    for entity_id, labels in enumerate(entity_labels):
        for label in labels:
            for rel, objs in dataset_one_hop[label]['relations'].items():
                for obj in objs:
                    if obj not in label2id:
                        continue
                    for obj_id in label2id[obj]:
                        if obj_id == entity_id:
                            continue
                        if obj_id in graph[entity_id][rel]:
                            continue
                        graph[entity_id][rel].append(obj_id)
    if all(not g for g in graph) and true_labels:
        raise Exception("Empty graph")
    return graph


def gen_dataset_graphs(filename, split, name):
    dataset = json.load(open(filename))
    dataset_labels = json.load(open(f"{split}_labels.json"))
    dataset_one_hop = json.load(open(f"{split}_one_hop.json"))
    graphs = []
    for docid, doc in tqdm(enumerate(dataset), desc=f"gen {name} data graph", ncols=100, total=len(dataset)):
        entities = doc['vertexSet']
        true_labels = doc['labels']
        entity_ids = [[] for _ in range(len(entities))]
        for k, entity in enumerate(entities):
            mention_names = list(set([mention['name'] for mention in entity]))
            all_set = [set() for _ in range(len(mention_names))]
            for i, mention_name in enumerate(mention_names):
                labels = dataset_labels[mention_name][:5]
                for label, _ in labels:
                    new_label = label[label.rfind("/") + 1:]
                    if not re.fullmatch(r'Q\d+', new_label):
                        continue
                    all_set[i].add(new_label)
            final_set = all_set[0]
            for i in range(1, len(all_set)):
                final_set = final_set | all_set[i]
            entity_ids[k].extend(list(final_set))
        graphs.append(gen_doc_graph(entity_ids, dataset_one_hop, true_labels))
    json.dump(graphs, open(f"./data/{name}_graph.json", "w"))
    return graphs


if __name__ == '__main__':
    # gen_dataset("./data/train_revised.json", "train")
    gen_dataset_one_hop_relations("train_labels.json", "test")
    # res = get_label("Gold by RIAA")
    gen_dataset_graphs("./data/train_revised.json", "dev", "test")
    x = 1
