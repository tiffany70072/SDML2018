# ICU prediction
## Challenge and Solution
* Missing value
  * d-GRU (Me)
  * Baseline: fill mean
* Irregularly sampled time
  * Take data in every fixed period
* Class imbalance
  * UnderSampling / OverSampling / Adversarial learning (Me)
  * Baseline: Class weight
* ICU data has various patterns
  * One class classification, Anomaly Detection in Time Series
* Length distribution
  * Resample training data
----
* Note: The team has three people, but only provide my code here. 
* TODO: Upload data_augmentation.py, data_augmentation_perturb.py, d-GRU.py
