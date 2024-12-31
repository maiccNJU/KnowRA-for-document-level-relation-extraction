import os
import shutil

import math
import json
import argparse
from collections import defaultdict
from copy import deepcopy
import logging

import rich
import torch
import numpy as np
from tqdm import tqdm
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from utils import set_seed
import hydra
from utils import get_lr, print_config_tree

log = logging.getLogger(__name__)


def train(cfg, datamodule, model):
    args = cfg.train
    if args.seed:  #  
        set_seed(args.seed)
    model.to(args.device)

    train_dataset, train_dataloader = datamodule.train_dataset, datamodule.train_dataloader()
    dev_dataset, dev_dataloader = datamodule.dev_dataset, datamodule.dev_dataloader()
    test_dataset, test_dataloader = datamodule.test_dataset, datamodule.test_dataloader()

    total_steps = args.epochs * (len(train_dataloader) // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"Total steps: {total_steps} = {args.epochs} epoch * ({len(train_dataloader)} batch // {args.gradient_accumulation_steps})")
    print(f"Warmup steps: {warmup_steps} = {total_steps} total steps * {args.warmup_ratio} warmup ratio")

    new_layer = ["extractor", "projection", "classifier", "conv"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)],
         "lr": args.classifier_lr},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.lr_schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = amp.GradScaler()

    num_steps = 0
    dev_best_score = -1
    test_best_score = -1
    model_name_or_path = cfg.model.model_name_or_path
    model_name_or_path = model_name_or_path[model_name_or_path.rfind("/") + 1:]
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for step, batch in enumerate(train_dataloader):
            model.train()
            inputs = {
                'input_ids': batch['input_ids'].to(args.device),
                'attention_mask': batch['attention_mask'].to(args.device),
                'hts': batch['hts'],
                'sent_pos': batch['sent_pos'],
                'entity_pos': batch['entity_pos'],
                'coref_pos': batch['coref_pos'],
                'mention_pos': batch['mention_pos'],
                'entity_types': batch['entity_types'],
                'men_graphs': batch['men_graphs'].to(args.device),
                'ent_graphs': batch['ent_graphs'].to(args.device),
                'etypes': batch['etypes'].to(args.device),
                'e_labels': batch['e_labels'].to(args.device),
                'labels': batch['labels'].to(args.device),
            }
            # with torch.autograd.set_detect_anomaly(True):
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = model(**inputs)
                if epoch == 0 and step == 0 and loss > 5.5:
                    print("initial loss: ", loss)
                    print("Bad loss, Stop Training ...")  # 
                    return
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                num_steps += 1
            if (args.log_steps > 0 and step % args.log_steps == 0) or (step + 1 == len(train_dataloader)):
                print(f"{epoch}/{step}/{len(train_dataloader)}: current loss {round(loss.item(), 4)}")
            if (step + 1) == len(train_dataloader) \
                    or (args.evaluation_steps > 0
                        and num_steps > total_steps // 2
                        and num_steps % args.evaluation_steps == 0
                        and step % args.gradient_accumulation_steps == 0
                        and num_steps > args.start_steps):
                dev_score, dev_output = evaluate(cfg, model, dev_dataset, dev_dataloader, tag="dev")
                print(dev_output)
                if epoch == 0 and (step + 1) == len(train_dataloader) and dev_score < 35:  # 
                    print("Bad result, Stop Training ...")
                    return
                lm_lr, classifier_lr = get_lr(optimizer)
                print(f'Current Step: {num_steps}, Current PLM lr: {lm_lr}, Current Classifier lr: {classifier_lr}')

                if dev_score > dev_best_score or dev_score > 60:
                    dev_best_score = dev_score
                    test_score, test_output = evaluate(cfg, model, test_dataset, test_dataloader, tag="test")
                    print(test_output)
                    # pred = report(args, model, test_features, collate_func)
                    # with open("result.json", "w") as fh:
                    #     json.dump(pred, fh)
                    if test_score > test_best_score:
                        test_best_score = test_score
                        save_dir = args.save_best_path
                        if not os.path.exists(save_dir):
                            os.mkdir(save_dir)
                        pre_max_model = [saved_model_name for saved_model_name in os.listdir(save_dir) if
                                         saved_model_name[:saved_model_name.find('_')] == model_name_or_path]
                        if len(pre_max_model) == 0:
                            pre_max_score = -1
                        else:
                            pre_max_score = max(float(saved_model_name[saved_model_name.rfind('_') + 1:])
                                                for saved_model_name in pre_max_model)
                        if args.save_best_path and test_score > pre_max_score:
                            sub_save_dir = f"{save_dir}/{model_name_or_path}_{round(test_score, 2)}"
                            save_model_path = f"{sub_save_dir}/docre_model.pth"
                            save_config_path = f"{sub_save_dir}/config.txt"
                            if not os.path.exists(sub_save_dir):
                                os.mkdir(sub_save_dir)
                            torch.save(model.state_dict(), save_model_path)
                            print_config_tree(cfg, open(save_config_path, "w"))
                            if pre_max_score != -1:
                                shutil.rmtree(f"{save_dir}/{model_name_or_path}_{pre_max_score}")
                if args.save_last_path:
                    torch.save(model.state_dict(), args.save_last_path)


def evaluate(cfg, model, dataset, dataloader, tag="dev"):
    assert tag in {"dev", "test"}
    args = cfg.train

    if tag == "dev":
        print("Evaluating")
    else:
        print("Testing")
    preds = []
    id2rel = dataset.id2rel
    rel_info = json.load(open("./data/Re-DocRED/meta/rel_info.json"))
    dataset_kg_scores = []

    model.to(args.device)
    for batch in dataloader:
        model.eval()

        inputs = {
            'input_ids': batch['input_ids'].to(args.device),
            'attention_mask': batch['attention_mask'].to(args.device),
            'hts': batch['hts'],
            'sent_pos': batch['sent_pos'],
            'entity_pos': batch['entity_pos'],
            'coref_pos': batch['coref_pos'],
            'mention_pos': batch['mention_pos'],
            'entity_types': batch['entity_types'],
            'men_graphs': batch['men_graphs'].to(args.device),
            'ent_graphs': batch['ent_graphs'].to(args.device),
            'etypes': batch['etypes'].to(args.device),
            'e_labels': None,
            'labels': None,
            'output_kg_scores': cfg.load_path is not None,  # 
        }

        with torch.no_grad():
            if cfg.load_path is not None and dataset.name == 're-docred':
                pred, e_scores = model(**inputs)
                e_scores = e_scores.tolist()
                etypes = batch['etypes'].tolist()
                e_labels = batch['e_labels'].int().tolist()
                i = 0
                for edge_hts in batch['edge_hts']:
                    adj = defaultdict(list)
                    for h, t in edge_hts:
                        rel = id2rel[etypes[i] + 1]
                        adj[h].append([t, rel, rel_info[rel], e_scores[i], e_labels[i]])
                        i += 1
                    dataset_kg_scores.append(adj)
            else:
                pred = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
    if dataset_kg_scores:
        json.dump(dataset_kg_scores, open(f"./case_study_{tag}.json", "w"))
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    if dataset.name == 'dwie':
        ign_f1, f1, p, r = dataset.official_evaluate(preds)
        output = {
            "F1": f1 * 100,
            "Ign_F1": ign_f1 * 100,
            "P": p * 100,
            "R": r * 100
        }
        return f1 * 100, output
    if tag == 'dev':
        re_f1, _, ign_f1, _, re_p, re_r, re_f1_freq, re_f1_long_tail, re_f1_intra, re_f1_inter, _, _, _, _ = \
            dataset.official_evaluate_benchmark(preds)
        output = {
            "dev_rel_F1": re_f1 * 100,
            "dev_ign_F1": ign_f1 * 100,
            "dev_P": re_p * 100,
            "dev_R": re_r * 100
        }
        return re_f1 * 100, output
    else:
        re_f1, _, ign_f1, _, re_p, re_r, re_f1_freq, re_f1_long_tail, re_f1_intra, re_f1_inter, _, _, _, _ = \
            dataset.official_evaluate_benchmark(preds)
        output = {
            "test_rel_F1": re_f1 * 100,
            "test_ign_F1": ign_f1 * 100,
            "test_Freq_F1": re_f1_freq * 100,
            "test_LT_F1": re_f1_long_tail * 100,
            "test_Intra_F1": re_f1_intra * 100,
            "test_Inter_F1": re_f1_inter * 100,
        }
        return re_f1 * 100, output


def report(args, model, features, collate_func):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_func,
                            drop_last=False)
    print("Predicting and save to ./result.json")
    preds = []

    for batch in dataloader:
        model.eval()

        inputs = {
            'input_ids': batch['input_ids'].to(args.device),
            'attention_mask': batch['attention_mask'].to(args.device),
            'hts': batch['hts'],
            'sent_pos': batch['sent_pos'],
            'entity_pos': batch['entity_pos'],
            'mention_pos': batch['mention_pos'],
            'entity_types': batch['entity_types'],
            'neg_mask': None,
            'sent_mask': None,
            'men_graphs': batch['men_graphs'].to(args.device),
            'ent_graphs': batch['ent_graphs'].to(args.device),
            'labels': None,
            'sent_labels': None
        }

        with torch.no_grad():
            pred = model(**inputs).cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    preds = to_official(preds, features)
    return preds


def get_random_mask(train_features, drop_prob):
    """
    
    """
    if drop_prob == 0:
        return train_features
    new_features = []
    n_e = 42
    for old_feature in tqdm(train_features, desc="random drop neg example"):
        feature = deepcopy(old_feature)
        #
        # tensor([1,1,0,1,0,1,1,1,...])
        neg_labels = torch.tensor(feature['labels'])[:, 0]
        # 
        #
        # tensor([0,1,3,5,6,7,...])
        neg_index = torch.where(neg_labels == 1)[0]
   
        # tensor([2,4,19,21,37,38,53,...])
        pos_index = torch.where(neg_labels == 0)[0]
        # 
        assert len(neg_index) + len(pos_index) == len(neg_labels)
        # 
        perm = torch.randperm(neg_index.size(0))
        # 
        sampled_negative_index = neg_index[perm[:int(drop_prob * len(neg_index))]]
        #
        neg_mask = torch.ones(len(feature['labels']))
        # 
        neg_mask[sampled_negative_index] = 0
        # feature['negative_mask'] = neg_mask
        # 
        # shape(42,42)
        pad_neg = torch.zeros((n_e, n_e))
        # 
        num_e = int(math.sqrt(len(neg_mask)))
        # 
        # pad_neg = [
        #     [1, 1, 1, ..., 0, 0, 0],
        #     [1, 0, 1, ..., 0, 0, 0],
        #     ...,
        #     [0, 0, 0, ..., 0, 0, 0],
        #     [0, 0, 0, ..., 0, 0, 0]
        # ] 42*42  1
        pad_neg[:num_e, :num_e] = neg_mask.view(num_e, num_e)
        feature['negative_mask'] = pad_neg
        new_features.append(feature)
    return new_features


def add_logits_to_features(features, logits):
    new_features = []
    for i, old_feature in enumerate(features):
        new_feature = deepcopy(old_feature)
        new_feature['teacher_logits'] = logits[i]
        assert logits[i].shape[0] == len(new_feature['hts'])
        new_features.append(new_feature)

    return new_features


def create_negative_mask(train_features, drop_prob):
    n_e = 42
    new_features = []
    for old_feature in train_features:
        feature = deepcopy(old_feature)
        neg_labels = np.array(feature['labels'])[:, 0]
        neg_index = np.squeeze(np.argwhere(neg_labels))
        sampled_negative_index = np.random.choice(neg_index, int(drop_prob * len(neg_index)))
        neg_mask = np.ones(len(feature['labels']))
        neg_mask[sampled_negative_index] = 0
        neg_mask = torch.tensor(neg_mask)
        pad_neg = torch.zeros((n_e, n_e))
        num_e = int(math.sqrt(len(neg_mask)))
        pad_neg[:num_e, :num_e] = neg_mask.view(num_e, num_e)
        feature['negative_mask'] = pad_neg
        new_features.append(feature)
    return new_features


@hydra.main(config_path="configs", config_name="train_docred.yaml", version_base="1.3")
def main(cfg):
    print_config_tree(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    log.info('Creating or Loading DataModule')
    datamodule = hydra.utils.instantiate(cfg.datamodule, tokenizer=tokenizer)()

    log.info("Creating DocRE Model")
    model = hydra.utils.instantiate(cfg.model, tokenizer=tokenizer)()

    if cfg.load_checkpoint:  # Training from checkpoint (for pre-training on distant dataset)
        log.info("Training from checkpoint")
        model.load_state_dict(torch.load(cfg.load_checkpoint))
        train(cfg, datamodule, model)
    elif cfg.load_path:  # Testing
        model.load_state_dict(torch.load(cfg.load_path))
        dev_score, dev_output = evaluate(cfg, model, datamodule.dev_dataset, datamodule.dev_dataloader(), tag="dev")
        print(dev_output)
        test_score, test_output = evaluate(cfg, model, datamodule.test_dataset, datamodule.test_dataloader(), tag="test")
        print(test_output)
    else:  # Training from scratch
        log.info("Training from scratch")
        train(cfg, datamodule, model)
    log.info("Finish Training or Testing")


if __name__ == "__main__":
    main()
