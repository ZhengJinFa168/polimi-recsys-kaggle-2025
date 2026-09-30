import numpy as np

from Recommenders.BaseRecommender import BaseRecommender


class ScoreBlendRecommender(BaseRecommender):
    """Weighted blend of the (normalized) scores of already-fitted recommenders.

    Fixes compared with the old DifferentLossRecommender:
      * normalization statistics ignore items the user already has (they are removed from the
        ranking anyway, and often carry the extreme scores that distorted min/max);
      * the user sample is seeded (results were different at every run);
      * scores are computed in one batch per recommender instead of one user at a time, and
        sum / l2 no longer recompute all the scores a second time;
      * a coefficient of 0 no longer produces NaN (0 * -inf) when items_to_compute is used;
      * no need to inherit from the item-similarity base class (there is no W_sparse here).
    """

    RECOMMENDER_NAME = "ScoreBlendRecommender"

    def __init__(self, URM_train, Recommenders, verbose=True, normalization_method="minmax",
                 n_users_sample=500, seed=1234):
        """
        normalization_method: "minmax" | "zscore" | "max" | "sum" | "l2" | None
        The recommenders must already be fitted (statistics are computed here).
        """
        super(ScoreBlendRecommender, self).__init__(URM_train, verbose=verbose)
        self.recommenders = Recommenders
        self.normalization_method = normalization_method
        self.n_users_sample = n_users_sample
        self.seed = seed
        self.coefficients = None
        self.normalization_stats = {}

        if normalization_method not in (None, "minmax", "zscore", "max", "sum", "l2"):
            raise ValueError("Unknown normalization method: {}".format(normalization_method))
        if normalization_method is not None:
            self._compute_normalization_stats()

    def fit(self, coefficients):
        if len(coefficients) != len(self.recommenders):
            raise ValueError("Number of coefficients ({}) doesn't match number of recommenders ({})".format(
                len(coefficients), len(self.recommenders)))
        self.coefficients = list(coefficients)

    def _compute_normalization_stats(self):
        self._print("Computing normalization statistics ({})...".format(self.normalization_method))
        rng = np.random.default_rng(self.seed)
        users = rng.choice(self.n_users, min(self.n_users_sample, self.n_users), replace=False)
        seen = self.URM_train[users].toarray() > 0

        for i, recommender in enumerate(self.recommenders):
            scores = np.asarray(recommender._compute_item_score(users), dtype=np.float64)
            scores[~np.isfinite(scores)] = 0.0
            candidate_scores = scores[~seen]                  # what the ranking actually looks at
            non_zero = candidate_scores[candidate_scores != 0]

            if len(non_zero) == 0:
                self.normalization_stats[i] = {"method": None}
                continue

            method = self.normalization_method
            if method == "minmax":
                stats = {"min": non_zero.min(), "max": non_zero.max()}
            elif method == "zscore":
                stats = {"mean": non_zero.mean(), "std": non_zero.std()}
            elif method == "max":
                stats = {"max": non_zero.max()}
            elif method == "sum":
                stats = {"avg_sum": np.where(seen, 0.0, scores).sum(axis=1).mean()}
            else:  # l2
                stats = {"avg_norm": np.linalg.norm(np.where(seen, 0.0, scores), axis=1).mean()}
            stats["method"] = method
            self.normalization_stats[i] = stats

    def _normalize_scores(self, scores, i):
        stats = self.normalization_stats.get(i, {"method": None})
        method = stats["method"]

        if method == "minmax":
            value_range = stats["max"] - stats["min"]
            return (scores - stats["min"]) / value_range if value_range > 0 else scores
        if method == "zscore":
            return (scores - stats["mean"]) / stats["std"] if stats["std"] > 0 else scores - stats["mean"]
        if method == "max":
            return scores / stats["max"] if stats["max"] > 0 else scores
        if method == "sum":
            return scores / stats["avg_sum"] if stats["avg_sum"] > 0 else scores
        if method == "l2":
            return scores / stats["avg_norm"] if stats["avg_norm"] > 0 else scores
        return scores

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if self.coefficients is None:
            raise RuntimeError("Call fit(coefficients) before recommending")
        user_id_array = np.atleast_1d(user_id_array)
        blended = np.zeros((len(user_id_array), self.n_items), dtype=np.float32)

        for i, (recommender, weight) in enumerate(zip(self.recommenders, self.coefficients)):
            if weight == 0:
                continue
            scores = np.asarray(recommender._compute_item_score(user_id_array, items_to_compute))
            scores = np.where(np.isfinite(scores), scores, 0.0)   # -inf masks are re-applied below
            blended += weight * self._normalize_scores(scores, i)

        if items_to_compute is not None:
            not_computed = np.ones(self.n_items, dtype=bool)
            not_computed[items_to_compute] = False
            blended[:, not_computed] = -np.inf
        return blended


# Backwards compatible name used by the old scripts
DifferentLossRecommender = ScoreBlendRecommender
