# Dependence note for the matched random comparator

Each random replicate partitions a newly drawn stratified pool of 430,000 transactions from the same 590,540-row IEEE-CIS labelled dataset. Because the requested pool is much larger than one quarter of the source dataset, transactions recur across replicate seeds. The replicate summaries are consequently correlated.

The manuscript uses these replicates to describe how the random-split estimand differs from strict later-period evaluation under matched sizes, prevalence, models, preprocessing, calibration, and alert budgets. It does not treat the four replicates as independent experimental units, does not report independent-sample hypothesis tests as confirmatory evidence, and does not interpret the observed gap as a pure causal leakage effect.
