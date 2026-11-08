# Evaluation protocol and the persisted full pipeline

Every Training Job uses the same protocol, so any two Runs are directly comparable.

- **Fixed Test Set.** An 80/20 split of the Dataset with `random_state=42`, set in settings and not in the Experiment Config. The Test Set is 3,303 rows.
- **Selection by CV only.** Each Candidate is scored with `KFold(5, shuffle=True, random_state=42)` on the training portion. The winner is the Candidate with the lowest mean CV RMSE. The Best Run is chosen the same way across Runs.
- **Test Set scored once.** The winner is refitted on the whole training portion and scored once on the Test Set. Test metrics are reported but never used to choose (notebook issue 8).
- **Original units.** RMSE, MAE and R² are computed on predictions in MedHouseVal units (100k USD), including for Runs with a log1p target (notebook issue 3).
- **The whole pipeline is persisted.** The optional `BedroomsPerRoom` step, the `StandardScaler` (linear regression only) and the estimator form one scikit-learn `Pipeline`, wrapped in `TransformedTargetRegressor(log1p, expm1)` when the target is transformed. It is fitted only on the rows it is trained on, and logged as one MLflow model whose signature comes from the shared schema module. Consumers send raw features and get predictions in original units, with nothing to re-implement at serving time (notebook issues 1, 2 and 4).

## Known limitation: geographic shift

The CSV is the first 16,512 rows of scikit-learn's California Housing dataset in their original order, not a random sample. The missing 4,128 rows skew towards Northern California (mean latitude 37.1 against 35.3, mean MedHouseVal 2.26 against 2.02). A random split and shuffled CV do not simulate that shift, so scores on those rows would likely be worse than our Test Set suggests. We treat the CSV as all the data we have, document the shift, and list geographic drift as a production monitoring signal instead of building spatial CV.

## Considered Options

- **Spatial or grouped CV by region**: closer to the shift above, but no region labels exist, and choosing a grouping is a modelling decision beyond this proof of concept.
- **Per-config split seeds**: flexible, but Runs with different seeds are not comparable on one Leaderboard.
- **Persist only the estimator and repeat feature steps in the API**: smaller artifacts, but training and serving code can drift apart, which is the class of bug the notebook had.
- **Grid search as a third tuning mode, next to fixed and random**: it is exhaustive, but a grid can only try a few hand-picked points of a continuous or log-scaled range, and its size is the product of the list lengths. The default gradient boosting space as a modest grid (5 × 5 × 3 × 4) is 300 Candidates, or 1,500 fits, for a single laptop worker that runs one job at a time. Random search takes those ranges directly, and `n_iter` (1 to 50) sets the cost, which the UI shows before launch. When only a few hyperparameters matter, random search also tries more distinct values of each one for the same budget (Bergstra & Bengio, 2012). If an exhaustive sweep is needed later, a `grid` mode that takes lists only, with Candidates as the product and the same cap of 50, fits the existing Candidate loop.

## Consequences

- Changing the split or folds in settings makes every earlier Run incomparable. They are logged on every Run (`split_seed`, `test_size`, `cv_folds`, `cv_seed`) together with `dataset_sha256`, so the Leaderboard can filter on them.
- The target is capped at 5.00001: 747 rows sit at or near the cap (724 at 5.00001, 23 at 5.0), so errors on the most expensive districts are understated. We do not correct for it.
- Every Run logs one child run per Candidate, even a fixed job with a single Candidate, so fixed jobs and random searches share one code path.
