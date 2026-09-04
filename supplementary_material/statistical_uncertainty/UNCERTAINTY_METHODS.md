# Statistical uncertainty analysis

No model was refitted. All calculations use frozen fold-level outputs.

- The temporal fold is the analysis unit. Sparkov repeated seeds are averaged within each future fold before uncertainty calculations.
- Mean sensitivity intervals use the exact finite empirical fold-resampling distribution: 46,656 ordered resamples for six Sparkov folds and 256 for four IEEE-CIS folds. These are descriptive sensitivity intervals over the observed blocks, not population-level confidence guarantees.
- Paired temporal comparisons use fold-specific differences, exact two-sided sign-flip tests, directional win counts, and standardized paired effect size `dz`. With four folds, the smallest attainable two-sided sign-flip p-value is 0.125.
- The four IEEE-CIS random replicates each use 430,000 of 590,540 labelled transactions, so their samples necessarily overlap. They are repeated benchmark realizations, not independent experimental units.
- Random-versus-temporal differences are therefore reported descriptively using all 65,536 combinations of the two exact four-block resampling distributions. Independent-sample permutation p-values and Hedges g are not used as confirmatory evidence.
- The random-temporal gap combines future-to-past mixing, closer train-test distribution matching under random assignment, and the distribution shift revealed by later-period testing; it is not a causal estimate of leakage alone.
