# polimi-recsys-kaggle-2025

Top-20 item recommendation for the **Recommender Systems course challenge at Politecnico di Milano (Kaggle, 2025)**.

## Task and data

Implicit-feedback recommendation: for each of 27,095 users, recommend 20 items they have not interacted with.

| | |
|---|---|
| Users / items | 27,095 / 6,969 |
| Interactions | 3,043,058 (density 1.6%) |
| Items per user | min 14, median 71 |
| Available signal | user-item interactions only (no content features, no timestamps) |

The data is not included.

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

`helping_methods.py` holds the shared code: data loading, the seeded split, evaluation printing, the model cache and a
submission writer that checks the file (20 distinct items per user) before saving.


## Credits

`Recommenders/`, `Evaluation/`, `Data_manager/`, `Utils/` and `HyperparameterTuning/` are the course framework by
Maurizio Ferrari Dacrema (Politecnico di Milano): https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi.
That code is not mine; the scripts at the top level are.
