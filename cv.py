import numpy as np
from sklearn.model_selection import StratifiedKFold


def build_folds(
    LOGGER,
    df,
    group_col="id",
    strate_col="categories",
    fold_col_name="fold",
    n_splits=5,
    seed=2222,
):
    LOGGER.info(f"Used seed: {seed}")
    df2 = df[[group_col, strate_col]].drop_duplicates([group_col]).copy(deep=False)
    df2[fold_col_name] = -1

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    splits = skf.split(np.arange(len(df2)), y=df2[strate_col])

    for fold, (train_idx, val_idx) in enumerate(splits):
        df2.loc[df2.index[val_idx], fold_col_name] = fold

    df = df.merge(df2[[group_col, fold_col_name]], on=group_col, how="left")

    return df
