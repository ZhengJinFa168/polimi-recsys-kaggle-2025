"""ItemKNN baseline with a TopPop fallback for users with very short profiles.

Replaces "Item Collaborative Filtering.py". The old script also wrote the submission with a model
trained on URM_train (80% of the data) instead of all of it.

    python item_knn_baseline.py
    python item_knn_baseline.py --submit
"""
import argparse

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop

from HybridRecommender import HybridRecommender
from helping_methods import CUTOFF, evaluate, load_urm, make_splits, toOutput

KNN_PARAMS = dict(shrink=10, topK=100)
MIN_RATINGS = 5


def build(URM):
    item_knn = ItemKNNCFRecommender(URM)
    item_knn.fit(**KNN_PARAMS)
    top_pop = TopPop(URM)
    top_pop.fit()
    return item_knn, HybridRecommender(item_knn, top_pop, URM, min_ratings_threshold=MIN_RATINGS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    URM_all, target_users = load_urm()
    splits = make_splits(URM_all)
    evaluator_validation = EvaluatorHoldout(splits.validation, cutoff_list=[CUTOFF])

    item_knn, hybrid = build(splits.train)
    evaluate(evaluator_validation, item_knn, "ItemKNN (validation)")
    evaluate(evaluator_validation, hybrid, "ItemKNN + TopPop fallback (validation)")

    if args.submit:
        _, final_hybrid = build(URM_all)
        toOutput(target_users, final_hybrid, "output_item_knn.csv")


if __name__ == "__main__":
    main()
