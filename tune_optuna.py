"""Optuna tuning for the course recommenders (replaces multiVAE.py and testing_with_optuna.py).

    python tune_optuna.py --model scaled_puresvd --trials 200
    python tune_optuna.py --model multvae --trials 20 --plots

Fixes compared with the old scripts:
  * the data was re-read and re-split (unseeded) inside every trial, so trials were scored on
    different splits; now one seeded split is built once;
  * tuning was done on the test split; now trials are scored on VALIDATION, and the untouched
    TEST split is used once at the end, on the best parameters refit on train + validation;
  * suggest_loguniform(lr, 1e-3, 1e-5) had low > high (invalid), and is deprecated: now
    suggest_float(..., log=True) with ordered bounds;
  * the tuned metric is METRIC from helping_methods (MAP by default), not hard-coded RECALL;
  * copy-pasted labels ("1 - Accuracy", study name "ITEMKNNRecommender" for a PureSVD study) are gone;
  * the sqlite folder is created if missing and studies resume (load_if_exists);
  * one file, one search space per model.

If you still have old .db files in OptunaStudies/, delete them: they were tuned on different splits.
"""
import argparse
import os

import optuna
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import ScaledPureSVDRecommender
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

from helping_methods import CUTOFF, METRIC, SEED, get_metric, load_urm, make_splits


# ---------------------------------------------------------------------------------------------
# Search spaces: each returns the full kwargs for recommender.fit(**params)
# ---------------------------------------------------------------------------------------------
def space_scaled_puresvd(trial):
    return dict(num_factors=trial.suggest_int("num_factors", 10, 1000),
                scaling_items=trial.suggest_float("scaling_items", 0.5, 1.5),
                scaling_users=trial.suggest_float("scaling_users", 0.5, 1.5),
                random_seed=1)


def space_item_knn(trial):
    return dict(topK=trial.suggest_int("topK", 10, 1000),
                shrink=trial.suggest_int("shrink", 0, 1000),
                similarity=trial.suggest_categorical("similarity", ["cosine", "jaccard", "dice"]),
                feature_weighting=trial.suggest_categorical("feature_weighting", ["none", "BM25", "TF-IDF"]))


def space_rp3beta(trial):
    return dict(topK=trial.suggest_int("topK", 10, 1000),
                alpha=trial.suggest_float("alpha", 0.1, 2.0),
                beta=trial.suggest_float("beta", 0.0, 2.0),
                normalize_similarity=trial.suggest_categorical("normalize_similarity", [True, False]))


def space_slim(trial):
    return dict(topK=trial.suggest_int("topK", 50, 1000),
                alpha=trial.suggest_float("alpha", 1e-4, 1e-1, log=True),
                l1_ratio=trial.suggest_float("l1_ratio", 1e-4, 0.5, log=True))


def space_ease_r(trial):
    return dict(l2_norm=trial.suggest_float("l2_norm", 1.0, 1e5, log=True),
                normalize_matrix=trial.suggest_categorical("normalize_matrix", [True, False]))


def space_ials(trial):
    return dict(num_factors=trial.suggest_int("num_factors", 20, 200),
                alpha=trial.suggest_float("alpha", 0.1, 10.0, log=True),
                epsilon=trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                reg=trial.suggest_float("reg", 1e-3, 10.0, log=True),
                init_mean=trial.suggest_float("init_mean", 0.0, 3.0),
                init_std=trial.suggest_float("init_std", 0.1, 3.0),
                confidence_scaling=trial.suggest_categorical("confidence_scaling", ["linear", "log"]))


def space_multvae(trial):
    return dict(epochs=trial.suggest_int("epochs", 50, 200),
                learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
                dropout=trial.suggest_float("dropout", 0.2, 0.7),
                total_anneal_steps=trial.suggest_int("total_anneal_steps", 50000, 300000),
                anneal_cap=trial.suggest_float("anneal_cap", 0.1, 0.5),
                l2_reg=trial.suggest_float("l2_reg", 1e-4, 1e-2, log=True),
                sgd_mode=trial.suggest_categorical("sgd_mode", ["adam", "sgd", "rmsprop"]),
                encoding_size=trial.suggest_int("encoding_size", 50, 200),
                next_layer_size_multiplier=trial.suggest_float("next_layer_size_multiplier", 1.5, 3.0),
                max_n_hidden_layers=trial.suggest_int("max_n_hidden_layers", 1, 3))


def make_multvae(URM):
    import torch  # imported lazily: only needed (and required) for --model multvae
    from Recommenders.Neural.MultVAE_PyTorch_Recommender import MultVAERecommender_PyTorch_OptimizerMask
    return MultVAERecommender_PyTorch_OptimizerMask(URM, use_gpu=torch.cuda.is_available())


MODELS = {
    "scaled_puresvd": (ScaledPureSVDRecommender, space_scaled_puresvd),
    "item_knn": (ItemKNNCFRecommender, space_item_knn),
    "rp3beta": (RP3betaRecommender, space_rp3beta),
    "slim": (SLIMElasticNetRecommender, space_slim),
    "ease_r": (EASE_R_Recommender, space_ease_r),
    "ials": (IALSRecommender, space_ials),
    "multvae": (make_multvae, space_multvae),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--plots", action="store_true", help="show optuna plots (needs a display)")
    args = parser.parse_args()

    factory, space = MODELS[args.model]
    URM_all, _ = load_urm()
    splits = make_splits(URM_all)                       # built ONCE, seeded
    evaluator_validation = EvaluatorHoldout(splits.validation, cutoff_list=[CUTOFF])
    evaluator_test = EvaluatorHoldout(splits.test, cutoff_list=[CUTOFF])

    def objective(trial):
        recommender = factory(splits.train)
        recommender.fit(**space(trial))
        results, _ = evaluator_validation.evaluateRecommender(recommender)
        return get_metric(results, CUTOFF, METRIC)

    os.makedirs("OptunaStudies", exist_ok=True)
    study = optuna.create_study(direction="maximize", study_name=args.model,
                                storage="sqlite:///OptunaStudies/{}.db".format(args.model),
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=args.trials)

    print("Finished trials:", len(study.trials))
    print("Best validation {}@{}: {:.5f}".format(METRIC, CUTOFF, study.best_value))
    print("Best params:")
    for key, value in study.best_params.items():
        print("    {}: {}".format(key, value))

    # One-shot check on the untouched test split: best params, refit on train + validation
    best_fit_params = space(optuna.trial.FixedTrial(study.best_params))
    final = factory(splits.train_complete)
    final.fit(**best_fit_params)
    results, _ = evaluator_test.evaluateRecommender(final)
    print("Held-out TEST {}@{}: {:.5f}".format(METRIC, CUTOFF, get_metric(results, CUTOFF, METRIC)))

    if args.plots:
        import optuna.visualization as vis
        for plot in (vis.plot_optimization_history, vis.plot_param_importances,
                     vis.plot_slice, vis.plot_parallel_coordinate):
            plot(study).show()


if __name__ == "__main__":
    main()
