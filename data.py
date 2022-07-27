import os
import re

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def get_cpc_texts(cpc_data_path):
    contexts = []
    pattern = "[A-Z]\d+"
    for file_name in os.listdir(f"{cpc_data_path}/CPCSchemeXML202105"):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]:
        with open(
            f"{cpc_data_path}/CPCTitleList202202/cpc-section-{cpc}_20220201.txt"
        ) as f:
            s = f.read()
        pattern = f"{cpc}\t\t.+"
        result = re.findall(pattern, s)
        cpc_result = result[0].lstrip(pattern)
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f"{context}\t\t.+"
            result = re.findall(pattern, s)
            results[context] = cpc_result + ". " + result[0].lstrip(pattern)
    return results


def prepare_input(cfg, text):
    inputs = cfg.tokenizer(
        text,
        add_special_tokens=True,
        max_length=cfg.max_len,
        padding="max_length",
        return_offsets_mapping=False,
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg

        df = df.join(df.groupby("anchor").target.agg(list).rename("ref"), on="anchor")
        df["ref2"] = df.apply(
            lambda x: [i for i in x["ref"] if i != x["target"]], axis=1
        )
        df["ref2"] = df.ref2.apply(
            lambda x: ", ".join(sorted(list(set(x))[0:30], key=x.index))
        )

        df["text"] = (
            df["anchor"]
            + "[SEP]"
            + df["target"]
            + "[SEP]"
            + df["context_text"]
            + "[SEP]"
            + df["ref2"]
        )

        self.texts = df["text"].values
        self.labels = df["score"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


class CustomCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        output = dict()

        output["input_ids"] = [sample[0]["input_ids"] for sample in samples]
        output["attention_mask"] = [sample[0]["attention_mask"] for sample in samples]

        labels = [sample[1] for sample in samples]

        # calculate max token length of this batch
        batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [
                s.tolist() + (batch_max - len(s)) * [self.tokenizer.pad_token_id]
                for s in output["input_ids"]
            ]
            output["attention_mask"] = [
                s.tolist() + (batch_max - len(s)) * [0]
                for s in output["attention_mask"]
            ]
        else:
            output["input_ids"] = [
                (batch_max - len(s)) * [self.tokenizer.pad_token_id] + s.tolist()
                for s in output["input_ids"]
            ]
            output["attention_mask"] = [
                (batch_max - len(s)) * [0] + s.tolist()
                for s in output["attention_mask"]
            ]

        output["input_ids"] = torch.tensor(output["input_ids"], dtype=torch.long)
        output["attention_mask"] = torch.tensor(
            output["attention_mask"], dtype=torch.long
        )

        labels = torch.tensor(labels, dtype=torch.float)

        return output, labels


def get_tokenizer(CFG, OUTPUT_DIR):
    if "deberta-v2" in CFG.model or "deberta-v3" in CFG.model:
        from transformers.models.deberta_v2 import DebertaV2TokenizerFast

        tokenizer = DebertaV2TokenizerFast.from_pretrained(
            CFG.model, trim_offsets=False
        )
        # special_tokens_dict = {'additional_special_tokens': ['[]']}
        # _ = tokenizer.add_special_tokens(special_tokens_dict)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(CFG.model, trim_offsets=False)

    tokenizer.save_pretrained(OUTPUT_DIR + "tokenizer/")
    CFG.tokenizer = tokenizer
    return tokenizer, CFG


def set_max_len(CFG, cpc_texts, tokenizer, train, LOGGER):
    lengths_dict = {}

    lengths = []
    tk0 = tqdm(cpc_texts.values(), total=len(cpc_texts))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(length)
    lengths_dict["context_text"] = lengths

    for text_col in ["anchor", "target"]:
        lengths = []
        tk0 = tqdm(train[text_col].fillna("").values, total=len(train))
        for text in tk0:
            length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            lengths.append(length)
        lengths_dict[text_col] = lengths

    CFG.max_len = (
        max(lengths_dict["anchor"])
        + max(lengths_dict["target"])
        + max(lengths_dict["context_text"])
        + 4
    )  # CLS + SEP + SEP + SEP
    LOGGER.info(f"max_len: {CFG.max_len}")

    return CFG
