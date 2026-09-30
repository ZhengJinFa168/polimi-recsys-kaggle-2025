import numpy as np
import scipy.sparse as sps

from Recommenders.BaseRecommender import BaseRecommender


class HybridRecommender(BaseRecommender):
    """Use `main_recommender` for users with enough interactions, `fallback_recommender` otherwise.

    Implemented through `_compute_item_score`, so BaseRecommender.recommend handles batching,
    seen-item removal, cutoff and single-user calls, and the course evaluator works unchanged.
    (The previous version overrode `recommend` with a per-user loop, printed "CIAO" for every cold
    user and did not support items_to_compute properly.)
    """

    RECOMMENDER_NAME = "ColdUserFallbackHybridRecommender"

    def __init__(self, main_recommender, fallback_recommender, URM_train, min_ratings_threshold=5):
        super(HybridRecommender, self).__init__(URM_train, verbose=False)
        self.main_recommender = main_recommender
        self.fallback_recommender = fallback_recommender
        self.min_ratings_threshold = min_ratings_threshold
        self.user_ratings_count = np.ediff1d(sps.csr_matrix(URM_train).indptr)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        user_id_array = np.atleast_1d(user_id_array)
        is_cold = self.user_ratings_count[user_id_array] < self.min_ratings_threshold

        scores = np.full((len(user_id_array), self.n_items), -np.inf, dtype=np.float32)
        if (~is_cold).any():
            scores[~is_cold] = self.main_recommender._compute_item_score(user_id_array[~is_cold], items_to_compute)
        if is_cold.any():
            scores[is_cold] = self.fallback_recommender._compute_item_score(user_id_array[is_cold], items_to_compute)
        return scores
