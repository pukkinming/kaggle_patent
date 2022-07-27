# Main entry of the codebase for Kaggle patent competition
# https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching

import os

import pandas as pd
import tokenizers
import torch
import transformers
from config import CFG
from cv import build_folds
from data import get_cpc_texts, get_tokenizer, set_max_len
from trainer import train_loop
from utils import get_logger, get_result, seed_everything

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
os.environ["TOKENIZERS_PARALLELISM"] = "true"

INPUT_DIR = "/ext_ssd/kaggle_patent/"
OUTPUT_DIR = "./"
CPC_DATA_PATH = "../../input/cpc-data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

LOGGER = get_logger()

LOGGER.info(f"torch.__version__: {torch.__version__}")
LOGGER.info(f"tokenizers.__version__: {tokenizers.__version__}")
LOGGER.info(f"transformers.__version__: {transformers.__version__}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if CFG.debug:
        CFG.epochs = 2
        CFG.trn_fold = [0]

    seed_everything(seed=CFG.seed)

    train = pd.read_csv(INPUT_DIR + "train.csv")
    test = pd.read_csv(INPUT_DIR + "test.csv")
    submission = pd.read_csv(INPUT_DIR + "sample_submission.csv")
    cpc_texts = get_cpc_texts(CPC_DATA_PATH)
    torch.save(cpc_texts, OUTPUT_DIR + "cpc_texts.pth")
    train["context_text"] = train["context"].map(cpc_texts).apply(lambda x: x.lower())
    train["text"] = (
        train["anchor"] + "[SEP]" + train["target"] + "[SEP]" + train["context_text"]
    )

    if CFG.debug:
        train = train.sample(n=1000, random_state=0).reset_index(drop=True)

    train = build_folds(
        LOGGER,
        train,
        group_col="anchor",
        strate_col="context",
        # fold_col_name=cfg.FOLD_COL_NAME,
        n_splits=5,
        seed=CFG.seed,
    )

    tokenizer, CFG = get_tokenizer(CFG, OUTPUT_DIR)
    CFG = set_max_len(CFG, cpc_texts, tokenizer, train, LOGGER)

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold, CFG, device, OUTPUT_DIR, LOGGER)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR + "oof_df.pkl")
