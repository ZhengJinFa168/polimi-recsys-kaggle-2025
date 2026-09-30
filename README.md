# polimi-recsys-kaggle-2025

Top-20 item recommendation for the **Recommender Systems course challenge at Politecnico di Milano (Kaggle, 2025)**.
This is a university course competition, **not** the ACM RecSys Challenge 2025.

## Task and data

Implicit-feedback recommendation: for each of 27,095 users, recommend 20 items they have not interacted with.

| | |
|---|---|
| Users / items | 27,095 / 6,969 |
| Interactions | 3,043,058 (density 1.6%) |
| Items per user | min 14, median 71 |
| Available signal | user-item interactions only (no content features, no timestamps) |

The data is not included. Put the competition files in `data/` as `data_train.csv` and `data_target_users_test.csv`.

## What is implemented

- **Item-based:** ItemKNN (cosine similarity, shrink 10, topK 100).
- **Linear / graph models:** SLIM-ElasticNet, RP3beta, EASE-R.
- **Matrix factorization:** IALS (tuned earlier; see status below).
- **Combination:** a similarity-matrix blend of SLIM and RP3beta; a normalized score-blend class (`MergeRecommender.py`); a fallback from a main model to TopPop for short profiles (`HybridRecommender.py`).
- **Two-stage ranking:** candidate generation followed by an XGBoost `XGBRanker` re-ranker.
- **Tuning:** Optuna search spaces for several models in `tune_optuna.py`.

## Evaluation protocol

All local numbers come from one seeded split of the interactions: **64% train / 16% validation / 20% test** (seed 1234).
Models are fit on train and scored on validation. The test split is used once at the end by `tune_optuna.py`.
The submission model is fit on all data. Recommendations are 20 items, seen items removed.

Two MAP definitions exist in the course framework and differ only in the denominator of average precision:
`MAP` divides by the list length (20); `MAP_MIN_DEN` divides by min(#relevant items, 20), the usual Kaggle "MAP@K".
`MAP_MIN_DEN` is always higher. Numbers under different definitions are never compared below.

## Results

Validation split, all users with at least one validation item:

| Model | MAP@20 | Recall@20 |
|---|---|---|
| ItemKNN (shrink 10, topK 100) | 0.0409 | 0.190 |
| SLIM-ElasticNet | **0.0545** | 0.244 |
| SLIM + RP3beta blend | 0.0545 | 0.244 |

- SLIM is about 33% better than ItemKNN.
- The RP3beta blend adds nothing measurable at the tuned weight (0.949 on SLIM).
- The TopPop fallback for short profiles never triggers here (every user has at least 14 interactions).

### XGBoost re-ranker: no gain over SLIM

Same 5,400 held-out users for every row; `MAP_MIN_DEN@20`. The ranker is trained on a different set of users.

| Candidate pool | Best single signal (SLIM scores) | XGBRanker | Change |
|---|---|---|---|
| ItemKNN top 30 (contains 16.9% of held-out items) | 0.0850 | 0.0853 | +0.3% |
| SLIM top 100 (contains 44.0% of held-out items) | 0.0967 | 0.0973 | +0.6% |

On plain MAP with the SLIM pool the ranker is slightly worse (0.05367 vs 0.05382).
An earlier comparison against the weak ItemKNN ordering alone looked like +21.7%, but that only reflected that the ranker
is given SLIM's scores as a feature. Against the best single signal there is no meaningful gain.

### Kaggle

Leaderboard score 0.50935, rank 40 of 71. That metric and holdout differ from the local validation numbers above,
so the two are not comparable; the local numbers are only for comparing models with each other.
The final submission pipeline (SLIM + RP3beta fit on all data) was rebuilt from scratch and reproduces the original
submission file: identical item lists, in identical order, for 99.88% of users.

## Running it

```
python item_knn_baseline.py    [--submit]
python hybrid_slim_rp3beta.py  [--submit]
python xgboost_reranker.py     [--generator slim --candidates 100] [--submit]
python train_ials.py           [--submit]
python tune_optuna.py --model {scaled_puresvd,item_knn,rp3beta,slim,ease_r,ials,multvae} --trials 50
```

Without `--submit` a script only prints validation scores. With it, the model is refit on all data and a CSV is written.
Fitting SLIM takes several minutes; fitted models are cached in `best_models_*/`, keyed by a hash of the hyperparameters.
`multvae` needs PyTorch and is only practical on a GPU.

`helping_methods.py` holds the shared code: data loading, the seeded split, evaluation printing, the model cache and a
submission writer that checks the file (20 distinct items per user) before saving.

## Status and limitations

- **Verified on the real data:** `item_knn_baseline.py`, `hybrid_slim_rp3beta.py`, `xgboost_reranker.py`.
- **Not re-run after the cleanup:** `train_ials.py` on the real data, MultVAE, and the Optuna search for every model except
  `item_knn` (a short smoke test). The tuned IALS initialization is large (`init_mean` 2.65, `init_std` 2.34) and a fit can diverge; the script
  checks for non-finite scores and stops instead of writing a broken file.
- **Optimistic validation numbers:** the SLIM and RP3beta hyperparameters were tuned earlier against random splits of the
  same data, so the scores above are somewhat optimistic. Single seed, no confidence intervals.
- **Old stack:** the vendored course framework targets an older Python/NumPy (it uses `np.int` and `np.in1d`, removed in
  recent NumPy). On a current NumPy it needs a small compatibility patch. The Cython extensions are optional; without them the
  framework falls back to slower Python code.

## Credits

`Recommenders/`, `Evaluation/`, `Data_manager/`, `Utils/` and `HyperparameterTuning/` are the course framework by
Maurizio Ferrari Dacrema (Politecnico di Milano): https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi.
That code is not mine; the scripts at the top level are.
