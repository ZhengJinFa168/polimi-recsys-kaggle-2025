"""SLIM-ElasticNet + RP3beta similarity blend.

Replaces fuckaroundHybrid.py. Fixes:
  * models were fit on URM_all and then evaluated on URM_test (a subset of URM_all): the score was
    measured on training data. Now: fit on TRAIN, evaluate on VALIDATION; only the submission uses URM_all.
  * the cached SLIM model was loaded regardless of which split it had been trained on. The cache is
    now keyed by hyperparameters and kept in a separate folder per split.

Note: these hyperparameters were tuned earlier against a random split of the same data, so the
validation number is still slightly optimistic. The Kaggle leaderboard is the real judge.

    python hybrid_slim_rp3beta.py
    python hybrid_slim_rp3beta.py --submit
"""
import argparse

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from helping_methods import CUTOFF, evaluate, fit_or_load, load_urm, make_splits, toOutput

SLIM_PARAMS = dict(topK=436, alpha=0.001239600142319664, l1_ratio=0.001002639662685697)
RP3_PARAMS = dict(alpha=1.7726359081010594, beta=0.3503980285071963, topK=48, normalize_similarity=True)
BLEND_ALPHA = 0.9487854330911072    # weight of SLIM in the blended similarity matrix
FINAL_TOPK = 994


def build_hybrid(URM, cache_folder):
    slim = fit_or_load(SLIMElasticNetRecommender(URM), cache_folder, "SLIM", **SLIM_PARAMS)
    rp3beta = RP3betaRecommender(URM)
    rp3beta.fit(**RP3_PARAMS)

    W_final = BLEND_ALPHA * slim.W_sparse + (1 - BLEND_ALPHA) * rp3beta.W_sparse
    hybrid = ItemKNNCustomSimilarityRecommender(URM)
    hybrid.fit(W_final, selectTopK=True, topK=FINAL_TOPK)
    return slim, hybrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    URM_all, target_users = load_urm()
    splits = make_splits(URM_all)
    evaluator_validation = EvaluatorHoldout(splits.validation, cutoff_list=[CUTOFF])

    slim, hybrid = build_hybrid(splits.train, "best_models_train/")
    evaluate(evaluator_validation, slim, "SLIM alone (validation)")
    evaluate(evaluator_validation, hybrid, "SLIM + RP3beta (validation)")

    if args.submit:
        _, final = build_hybrid(URM_all, "best_models_full/")
        toOutput(target_users, final, "output_hybrid_slim_rp3beta.csv")


if __name__ == "__main__":
    main()
