from torch.utils.data import Dataset
import torch
import transformers as ppb
import json
import numpy as np
from typing import List, Dict
import os
import sys

max_ent = 42
max_lbl = 151


class DocRED(Dataset):
    def __init__(self, data_path):
        tokenizer_class, pretrained_weights = (ppb.BertTokenizer,
                                               'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights,
                                                         padding_side="right",
                                                         pad_token="[PAD]"
                                                         )

        special_tokens_dict = {'additional_special_tokens': ['<e>', '</e>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.embedding_model = ppb.BertModel.from_pretrained("bert-base-uncased")
        self.embedding_model.resize_token_embeddings(self.get_token_embedding())

        self.data, self.rel2id = self.__read_data__(data_path)

    def __len__(self):
        return len(self.data)

    def __read_data__(self, data_dir: "str"):
        with open(data_dir, "r") as file:
            data = json.loads(file.read())

        rel2id_dir = "/".join(data_dir.split("/")[:-1] + ["rel2id.json"])
        with open(rel2id_dir, "r") as file:
            rel2id = json.loads(file.read())

        return data, rel2id

    def __add_special_entity_token__(self, doc: Dict):
        """
        Add special token at the start and the ending of the entity mentions
        and fix the positions
        :return: Fixed position data
        """
        # shift for every entity in a sentence as we proceed to add special tokens
        # one by one in a sentence
        entities = {key: [] for key in range(len(doc["sents"]))}
        for ent_idx, entity in enumerate(doc["vertexSet"]):
            for mnt_idx, mention in enumerate(entity):
                entities[mention["sent_id"]].append({"mention": mention,
                                                     "ent_idx": ent_idx,
                                                     "mnt_idx": mnt_idx})
        for sent_id, mention_list in entities.items():
            shift = 0
            mention_list = sorted(mention_list, key=lambda x: x["mention"]["pos"][0])
            for mnt_dic in mention_list:
                mention = mnt_dic["mention"]
                ent_idx = mnt_dic["ent_idx"]
                mnt_idx = mnt_dic["mnt_idx"]
                start, end = mention["pos"]

                start += shift
                end += shift

                sent = doc["sents"][sent_id]
                sent = sent[:start] + ["<e>"] + sent[start:end] + ["</e>"] + sent[end:]

                doc["sents"][sent_id] = sent
                doc["vertexSet"][ent_idx][mnt_idx]["pos"] = [start, end + 2]
                shift += 2

        return doc

    def __join_sents__(self, doc: Dict):
        len_sents = [len(sent) for sent in doc["sents"]]
        for ent in doc["vertexSet"]:
            for mnt in ent:
                mnt["pos"][0] += sum(len_sents[:mnt["sent_id"]])
                mnt["pos"][1] += sum(len_sents[:mnt["sent_id"]])

        doc["sents"] = sum(doc["sents"], [])
        doc["sents"] = list(map(lambda x: x.lower(), doc["sents"]))
        return doc

    def __tokenize__(self, doc: List[str]):
        input_ids = self.tokenizer.encode(doc[:510],
                                          add_special_tokens=True,
                                          padding="max_length",
                                          max_length=512,
                                          return_tensors="pt")
        attention_mask = input_ids.gt(0)
        return input_ids, attention_mask

    def __preprocess__(self, doc):
        doc = self.__add_special_entity_token__(doc)
        doc = self.__join_sents__(doc)
        tokenized_doc, attention_mask = self.__tokenize__(doc["sents"])

        embedded_doc = self.embedding_model(
            input_ids=torch.squeeze(tokenized_doc, 1),
            attention_mask=torch.squeeze(attention_mask, 1),
            output_attentions=True)

        vertexSet = [[mnt["pos"] for mnt in ent] for ent in doc["vertexSet"]]
        entity_list = self.aggregate_entities(vertexSet, embedded_doc.last_hidden_state.squeeze())
        labels = torch.zeros(42, 42, 97)
        for triple in doc["labels"]:
            i = triple["h"]
            j = triple["t"]
            k = self.rel2id[triple["r"]]
            labels[i][j][k] = 1

        processed_doc = {
            "entity_list": entity_list,
            "embedded_doc": embedded_doc,
            "label": labels
        }
        return processed_doc

    def aggregate_entities(self, vertexSet: list, embedded_doc: torch.Tensor) -> torch.Tensor:
        """
        :param vertexSet: list of mentions positions in a doc in shape (max_entity, max_mentions, 2)
        :param embedded_doc: embedded doc in shape (512, 768)
        :return: The logsumexp pooling of mentions for each entity in shape(max_entity, 768)
        """
        logsumexp = lambda x: torch.log(torch.sum(torch.exp(x), axis=0)).detach().numpy()
        _, embedding_size = embedded_doc.size()
        padded_result = torch.zeros(max_ent, embedding_size)
        mention_positions = [list(filter(lambda x: x[0] <= 512 and x[1] <= 512, ent)) for ent in vertexSet]
        mention_positions = [list(filter(lambda x: len(x) > 0, mnt)) for mnt in mention_positions]
        mention_positions = list(filter(lambda x: len(x) > 0, mention_positions))
        #  doc contains aggregated entity embedding, in shape(ent, 768)
        doc = [logsumexp(torch.concat([embedded_doc[pos[0]:pos[1]] for pos in mnt]))
               for mnt in mention_positions]
        # padding the doc to the length max_entity
        padded_result[:len(doc)] = torch.from_numpy(np.array(doc)).to(padded_result)
        return padded_result

    def __getitem__(self, idx):
        item = self.__preprocess__(self.data[idx])
        return item

    def get_token_embedding(self):
        return len(self.tokenizer)
