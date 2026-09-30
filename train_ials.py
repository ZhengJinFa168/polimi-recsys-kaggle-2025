"""IALS: honest validation score, then (optionally) refit on all data and write the submission.

Replaces fuckaroud.py, which fit on URM_all and then evaluated on a split of URM_all (so the
reported number was measured on data the model had trained on), and saved the IALS model under
the name "bestMULTIVAERecommender".

    python train_ials.py            # validation score only
    python train_ials.py --submit   # also refit on everything and write output_ials.csv
"""
import argparse
import time

import numpy as np

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

from helping_methods import CUTOFF, SEED, evaluate, load_urm, make_splits, toOutput

IALS_PARAMS = dict(num_factors=86, alpha=3.17051677957448, epsilon=0.039873271755949916,
                   reg=0.7743602221283774, init_mean=2.6486507999035966, init_std=2.3446108787910958,
                   confidence_scaling="log")


def fit_ials(URM):
    """Seeded fit + divergence check.

    These tuned parameters use a large random initialisation (init_mean ~2.6, init_std ~2.3) and the
    initialisation is not seeded, so a fit can occasionally blow up to inf/NaN. That would silently
    produce a broken submission (short or arbitrary recommendation lists), so fail loudly instead.
    """
    np.random.seed(SEED)
    recommender = IALSRecommender(URM)
    recommender.fit(**IALS_PARAMS)
    scores = recommender._compute_item_score(np.arange(min(200, URM.shape[0])))
    if not np.isfinite(scores).all():
        raise RuntimeError("IALS diverged (non-finite scores). Try another seed or a smaller init_std / init_mean.")
    return recommender


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    URM_all, target_users = load_urm()
    splits = make_splits(URM_all)
    evaluator_validation = EvaluatorHoldout(splits.validation, cutoff_list=[CUTOFF])

    start = time.time()
    recommender = fit_ials(splits.train)                 # fit on TRAIN only ...
    evaluate(evaluator_validation, recommender, "IALS (validation)")   # ... scored on unseen data
    print("Fit + evaluation took {:.1f}s".format(time.time() - start))

    if args.submit:
        final = fit_ials(URM_all)                         # final model uses every interaction
        final.save_model(folder_path="best_models_full/", file_name="IALS_final")
        toOutput(target_users, final, "output_ials.csv")


if __name__ == "__main__":
    main()
