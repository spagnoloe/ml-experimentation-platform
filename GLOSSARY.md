# Experimentation Platform

A platform where a data scientist trains and compares models that predict California district house values, and chooses which model other teams and systems get predictions from.

## Data

**Dataset**:
The provided CSV of California districts with eight features and the target, identified by a hash of its content.
_Avoid_: Training data (when the whole file is meant), data file

**Test Set**:
The fixed 20% of the Dataset held out from every Training Job and never used to choose between Candidates or Runs.
_Avoid_: Validation set, holdout

## Training

**Experiment Config**:
The data scientist's description of what to train: algorithm, feature options, target transform, and either fixed hyperparameters or a search space.
_Avoid_: Experiment, job spec

**Training Job**:
One queued or running execution of an Experiment Config.
_Avoid_: Experiment, task

**Candidate**:
One hyperparameter set evaluated with cross-validation inside a Training Job; a fixed-hyperparameter job has exactly one.
_Avoid_: Trial, child run

**Run**:
The tracked record of one Training Job; once completed, it holds the winning Candidate refitted on the training portion, scored once on the Test Set, and its fitted pipeline.
_Avoid_: Parent run, experiment, model

**Best Run**:
The completed Run with the lowest mean cross-validation RMSE on the current Dataset.
_Avoid_: Champion, winner

**Baseline**:
The top-ranked Run of the `dummy` algorithm, a mean predictor with no hyperparameters, shown on the Leaderboard as the reference every model should beat.
_Avoid_: Benchmark, dummy model

**Leaderboard**:
All completed Runs on the current Dataset, ranked by mean cross-validation RMSE.

## Models and serving

**Registered Model**:
A named line of Model Versions that all answer the same prediction question.
_Avoid_: Model (on its own), project

**Model Version**:
An immutable, numbered entry in a Registered Model, created by registering a Run; only Model Versions can be deployed.
_Avoid_: Model, artifact, pickle

**Deployment**:
A recorded decision that a specific Model Version serves predictions from now on; Deployments are never changed or removed.
_Avoid_: Release, promotion, alias, rollback

**Active Model**:
The Model Version named by the most recent Deployment.
_Avoid_: Champion, production model, current model
