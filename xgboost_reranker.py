"""Two-stage recommender: ItemKNN candidate generator + XGBRanker re-ranker.

Replaces XGBoost.py, which never evaluated the ranker, used raw UserID / ItemID as features
(lets the model memorise ids) and filled the feature table with a quadratic per-user `.loc` loop.

Protocol (no leakage):
  1. base models are fit on URM_train only; candidates and features come from URM_train only
  2. label = the candidate appears in URM_validation
  3. users with validation items are split 80/20: the ranker is trained on the first group and
     compared with the plain candidate-generator order on the second group (same MAP@CUTOFF)
  4. --submit: retrain the ranker on all labelled users, refit the base models on URM_all,
     re-rank the target users' candidates and write the submission

    python xgboost_reranker.py
    python xgboost_reranker.py --submit
"""
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm
from xgboost import XGBRanker

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from helping_methods import CUTOFF, SEED, fit_or_load, load_urm, make_splits, save_submission

N_CANDIDATES = 30
SLIM_PARAMS = dict(topK=436, alpha=0.001239600142319664, l1_ratio=0.001002639662685697)
EASE_PARAMS = dict(topK=100, l2_norm=1e3, normalize_matrix=False)
RANKER_PARAMS = dict(objective="rank:pairwise", n_estimators=50, learning_rate=0.1, reg_alpha=0.1,
                     reg_lambda=0.1, max_depth=5, grow_policy="depthwise", booster="gbtree",
                     random_state=SEED, verbosity=0)


def build_base_models(URM, cache_folder, generator_name="itemknn"):
    """Returns (candidate_generator, {feature_name: fitted_recommender}).

    generator_name: "itemknn" (default ItemKNN, cheap but weak) or "slim" (much stronger candidate pool).
    """
    item_knn = ItemKNNCFRecommender(URM, verbose=False)
    item_knn.fit()
    top_pop = TopPop(URM)
    top_pop.fit()
    p3alpha = P3alphaRecommender(URM, verbose=False)
    p3alpha.fit()
    slim = fit_or_load(SLIMElasticNetRecommender(URM), cache_folder, "SLIM", **SLIM_PARAMS)
    ease = fit_or_load(EASE_R_Recommender(URM), cache_folder, "EASE_R", **EASE_PARAMS)
    models = {"ItemKNN": item_knn, "TopPop": top_pop, "P3alpha": p3alpha, "SLIM": slim, "EASE_R": ease}
    generator = {"itemknn": item_knn, "slim": slim}[generator_name]
    return generator, models


def build_feature_frame(URM, generator, models, users, n_candidates=N_CANDIDATES, batch_size=256):
    """One row per (user, candidate item), users in the given order, candidates in generator order."""
    URM = sps.csr_matrix(URM)
    users = np.asarray(users)
    item_popularity = np.ediff1d(sps.csc_matrix(URM).indptr)
    user_profile_len = np.ediff1d(URM.indptr)

    frames = []
    for start in tqdm(range(0, len(users), batch_size), desc="features"):
        batch = users[start:start + batch_size]
        candidates = np.asarray(generator.recommend(batch, cutoff=n_candidates, remove_seen_flag=True))
        rows = np.arange(len(batch))[:, None]

        data = {"UserID": np.repeat(batch, n_candidates),
                "ItemID": candidates.ravel(),
                "generator_rank": np.tile(np.arange(n_candidates), len(batch))}
        for name, model in models.items():
            scores = model._compute_item_score(batch)
            data[name] = scores[rows, candidates].ravel()
        data["item_popularity"] = item_popularity[candidates.ravel()]
        data["user_profile_len"] = np.repeat(user_profile_len[batch], n_candidates)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


FEATURES = ["generator_rank", "ItemKNN", "TopPop", "P3alpha", "SLIM", "EASE_R", "item_popularity", "user_profile_len"]


def add_labels(frame, URM_truth):
    coo = sps.coo_matrix(URM_truth)
    truth = pd.DataFrame({"UserID": coo.row, "ItemID": coo.col, "Label": 1})
    frame = frame.merge(truth, on=["UserID", "ItemID"], how="left")     # left join keeps row order
    frame["Label"] = frame["Label"].fillna(0).astype(int)
    return frame


def train_ranker(frame):
    """Users without any positive candidate carry no ranking signal, so they are dropped."""
    frame = frame[frame.groupby("UserID")["Label"].transform("max") > 0]
    frame = frame.sort_values(["UserID", "generator_rank"], kind="stable")   # qid must be contiguous
    ranker = XGBRanker(**RANKER_PARAMS)
    ranker.fit(frame[FEATURES], frame["Label"], qid=frame["UserID"])
    return ranker


def mean_ap(frame, score_column, URM_truth, k=CUTOFF):
    """Returns (MAP, MAP_MIN_DEN) of the top-k of `score_column`, averaged over the users in `frame`.

    Same two definitions as the course evaluator (see helping_methods.evaluate):
    MAP divides by k, MAP_MIN_DEN divides by min(#relevant, k).
    """
    URM_truth = sps.csr_matrix(URM_truth)
    aps, aps_min_den = [], []
    for user, group in frame.groupby("UserID", sort=False):
        top_items = group.nlargest(k, score_column)["ItemID"].to_numpy()
        relevant = URM_truth.indices[URM_truth.indptr[user]:URM_truth.indptr[user + 1]]
        is_relevant = np.isin(top_items, relevant, assume_unique=True)
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float64) / (1 + np.arange(len(is_relevant)))
        aps.append(p_at_k.sum() / k)
        aps_min_den.append(p_at_k.sum() / min(len(relevant), k))
    return float(np.mean(aps)), float(np.mean(aps_min_den))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--generator", choices=["itemknn", "slim"], default="itemknn")
    parser.add_argument("--candidates", type=int, default=N_CANDIDATES,
                        help="candidates per user handed to the ranker (must be >= list length %d)" % CUTOFF)
    args = parser.parse_args()
    if args.candidates < CUTOFF:
        parser.error("--candidates must be >= %d" % CUTOFF)

    URM_all, target_users = load_urm()
    splits = make_splits(URM_all)

    print("Fitting base models on URM_train ...")
    generator, models = build_base_models(splits.train, "best_models_train/", args.generator)

    labelled_users = np.where(np.ediff1d(sps.csr_matrix(splits.validation).indptr) > 0)[0]
    frame = build_feature_frame(splits.train, generator, models, labelled_users, n_candidates=args.candidates)
    frame = add_labels(frame, splits.validation)

    # Split USERS (not rows) so the ranker is scored on users it has never seen
    permuted = np.random.default_rng(SEED).permutation(labelled_users)
    cut = int(0.8 * len(permuted))
    train_users, eval_users = permuted[:cut], permuted[cut:]

    ranker = train_ranker(frame[frame["UserID"].isin(train_users)])
    eval_frame = frame[frame["UserID"].isin(eval_users)].copy()
    eval_frame["ranker_score"] = ranker.predict(eval_frame[FEATURES])
    eval_frame["generator_order"] = -eval_frame["generator_rank"]

    # Same 30 candidates per user, ordered by each single signal and by the ranker. The fair question
    # is whether the ranker beats the BEST single signal, not the (weak) candidate generator alone.
    orderings = [("candidate generator order ({})".format(args.generator), "generator_order")] + \
                [("{} scores only".format(name), name) for name in ("SLIM", "EASE_R", "P3alpha", "TopPop")] + \
                [("XGBRanker re-ranking", "ranker_score")]
    print("Held-out users: {} | candidates per user: {} | list length: {}".format(
        len(eval_users), args.candidates, CUTOFF))
    print("{:<46s} {:>10s} {:>13s}".format("ordering of the same candidates", "MAP@20", "MAP_MIN_DEN@20"))
    scores = {}
    for label, column in orderings:
        scores[column] = mean_ap(eval_frame, column, splits.validation)
        print("{:<46s} {:>10.5f} {:>13.5f}".format(label, *scores[column]))

    best_single = max(("SLIM", "EASE_R", "P3alpha", "TopPop", "generator_order"), key=lambda c: scores[c][1])
    gain = scores["ranker_score"][1] / scores[best_single][1] - 1
    print("Ranker vs best single signal ({}): {:+.1%} relative (MAP_MIN_DEN@20)".format(best_single, gain))
    n_relevant = int(np.ediff1d(sps.csr_matrix(splits.validation).indptr)[eval_users].sum())
    print("Candidate pool contains {:.1%} of the held-out items: this caps what any re-ranking can reach.".format(
        eval_frame["Label"].sum() / n_relevant))
    print("Feature importance:", dict(zip(FEATURES, np.round(ranker.feature_importances_, 3))))

    if args.submit:
        print("Final ranker on all labelled users, base models on URM_all ...")
        final_ranker = train_ranker(frame)
        generator_full, models_full = build_base_models(URM_all, "best_models_full/", args.generator)
        target_frame = build_feature_frame(URM_all, generator_full, models_full, target_users,
                                           n_candidates=args.candidates)
        target_frame["ranker_score"] = final_ranker.predict(target_frame[FEATURES])

        top = (target_frame.sort_values(["UserID", "ranker_score"], ascending=[True, False])
               .groupby("UserID").head(CUTOFF))
        item_lists = top.groupby("UserID")["ItemID"].apply(list).reindex(target_users)
        save_submission(target_users, item_lists.tolist(), "output_xgboost_reranker.csv")


if __name__ == "__main__":
    main()
