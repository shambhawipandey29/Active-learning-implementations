# Active-learning-implementations
Python files for implimenting active learning and a dataset folder.

There are implemenations for three active learning query straegies:
1. max disagreemet within committee
2. Expected error (train)
3. expected error (test)
Expected error (test) uses dummy learner to reduce computational time. The final result with/without dummy learner is the same.


The datset folder has the following files:
1. LMTO_B_new.csv is the Perovskite data file with LMTO based geometric features
2. train.csv is directly adaopted from the study by [Li et. al](https://github.com/zhengl0217/Perovskite-Electronic-Structure-Feature-Python). It is the Perovskite data file with DFT features
