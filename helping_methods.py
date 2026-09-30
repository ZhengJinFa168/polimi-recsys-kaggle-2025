"""Shared helpers for the RecSys challenge scripts.

Everything the scripts had copy-pasted (data loading, splits, evaluation, output) lives here.
"""
import hashlib
import json
import os
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample

# ---------------------------------------------------------------------------------------------
# Config: check CUTOFF / METRIC against the competition's evaluation page.
# ---------------------------------------------------------------------------------------------
SEED = 1234
CUTOFF = 20          # number of recommendations per user (also the submission length)
METRIC = "MAP"       # metric used for tuning / model selection
DATA_TRAIN = "data/data_train.csv"
DATA_TARGET = "data/data_target_users_test.csv"

Splits = namedtuple("Splits", ["train", "validation", "test", "train_complete"])


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def load_urm(train_path=DATA_TRAIN, target_path=DATA_TARGET):
    """Return (URM_all, target_user_list).

    - The shape is explicit and covers every target user, so a target user that is missing from
      the training file (cold user) does not crash `recommend` later.
    - Duplicated (user, item) rows are collapsed: the URM is strictly binary.
    """
    df = pd.read_csv(train_path)
    df.columns = ["UserID", "ItemID"]
    target_users = pd.read_csv(target_path)["user_id"].to_numpy()

    n_users = int(max(df["UserID"].max(), target_users.max())) + 1
    n_items = int(df["ItemID"].max()) + 1

    URM_all = sps.coo_matrix(
        (np.ones(len(df), dtype=np.float32), (df["UserID"].values, df["ItemID"].values)),
        shape=(n_users, n_items)).tocsr()
    URM_all.sum_duplicates()
    URM_all.data[:] = 1.0
    return URM_all, target_users.tolist()


def make_splits(URM_all, seed=SEED):
    """Reproducible 64% train / 16% validation / 20% test split.

    - tune hyperparameters on `validation` (models fit on `train`)
    - touch `test` ONCE at the end (models fit on `train_complete` = train + validation)
    - the submission is fit on `URM_all`
    """
    np.random.seed(seed)  # the course split function uses the global numpy RNG
    URM_train_complete, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_complete, train_percentage=0.80)
    return Splits(URM_train, URM_validation, URM_test, URM_train_complete)


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------
def get_metric(results, cutoff=CUTOFF, metric=METRIC):
    """Extract one number from EvaluatorHoldout output (DataFrame indexed by cutoff, or a dict)."""
    if isinstance(results, pd.DataFrame):
        return float(results.loc[cutoff, metric])
    return float(results[cutoff][metric])


def evaluate(evaluator, recommender, name=None, cutoff=CUTOFF):
    """Evaluate and print MAP / MAP_MIN_DEN / Recall. Returns the value of METRIC.

    The two MAP flavours differ ONLY in the denominator of average precision:
      MAP          divides by the list length (@cutoff)            -> the course evaluator's default
      MAP_MIN_DEN  divides by min(#relevant items, @cutoff)        -> the usual Kaggle "MAP@K"
    MAP_MIN_DEN is always the larger one. Never compare numbers across the two definitions.
    """
    results, _ = evaluator.evaluateRecommender(recommender)
    name = name or recommender.RECOMMENDER_NAME
    print("{:<32s} MAP@{c} = {:.5f} | MAP_MIN_DEN@{c} = {:.5f} | Recall@{c} = {:.5f}".format(
        name, get_metric(results, cutoff, "MAP"), get_metric(results, cutoff, "MAP_MIN_DEN"),
        get_metric(results, cutoff, "RECALL"), c=cutoff))
    return get_metric(results, cutoff, METRIC)


def precision(recommended_items, relevant_items):
    is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)
    return np.sum(is_relevant, dtype=np.float32) / len(is_relevant)


def recall(recommended_items, relevant_items):
    is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)
    return np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]


def AP(recommended_items, relevant_items):
    """Average precision, same definition as the course evaluator (divides by min(#relevant, cutoff))."""
    is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    return np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])


def evaluate_algorithm(URM_test, recommender_object, at=5):
    """Slow reference implementation (the course EvaluatorHoldout is the one to use)."""
    URM_test = sps.csr_matrix(URM_test)
    cumulative_precision = cumulative_recall = cumulative_AP = 0.0
    num_eval = 0

    for user_id in range(URM_test.shape[0]):
        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]
        if len(relevant_items) > 0:
            recommended_items = np.asarray(recommender_object.recommend(user_id, cutoff=at))  # was `at=at`: no such argument
            num_eval += 1
            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_AP += AP(recommended_items, relevant_items)

    print("Recommender results are: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(
        cumulative_precision / num_eval, cumulative_recall / num_eval, cumulative_AP / num_eval))


# ---------------------------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------------------------
def fit_or_load(recommender, folder, name, **fit_kwargs):
    """Fit `recommender` or load it from `folder` if the same model was already trained.

    The file name contains a hash of the hyperparameters, and each split must use its own
    folder (e.g. best_models_train/ vs best_models_full/). This prevents the old bug where a
    model trained on one split was silently reloaded and used for another.
    """
    folder = folder.rstrip("/") + "/"
    os.makedirs(folder, exist_ok=True)
    tag = hashlib.md5(json.dumps(fit_kwargs, sort_keys=True, default=str).encode()).hexdigest()[:8]
    file_name = "{}_{}".format(name, tag)

    if os.path.exists(folder + file_name + ".zip"):
        print("{} already trained, loading {}".format(name, folder + file_name))
        recommender.load_model(folder_path=folder, file_name=file_name)
    else:
        recommender.fit(**fit_kwargs)
        recommender.save_model(folder_path=folder, file_name=file_name)
    return recommender


# ---------------------------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------------------------
def save_submission(user_ids, item_lists, output_file, cutoff=CUTOFF):
    """Write the `user_id,item_list` csv after checking it is well formed."""
    bad = [u for u, items in zip(user_ids, item_lists) if len(items) != cutoff or len(set(items)) != cutoff]
    if bad:
        raise ValueError("{} users do not have exactly {} distinct items (first: {})".format(len(bad), cutoff, bad[:5]))

    pd.DataFrame({"user_id": [int(u) for u in user_ids],
                  "item_list": [" ".join(str(int(i)) for i in items) for items in item_lists]}
                 ).to_csv(output_file, index=False)
    print("Results saved to " + output_file)


def toOutput(users_to_test, recommender, output_file, cutoff=CUTOFF, batch_size=1000):
    """Recommend `cutoff` unseen items to every target user and write the submission.

    Batched (the old version appended a row to a DataFrame per user, quadratic in the number of users).
    """
    users = np.asarray(users_to_test)
    item_lists = []
    for start in range(0, len(users), batch_size):
        item_lists.extend(recommender.recommend(users[start:start + batch_size], cutoff=cutoff, remove_seen_flag=True))
    save_submission(users, item_lists, output_file, cutoff)
