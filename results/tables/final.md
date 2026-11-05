# Final comparison — 550 labels, mean ± sd over 20 seeds

|                   | oracle        | naive HITL       | final system     |
|:------------------|:--------------|:-----------------|:-----------------|
| balanced_accuracy | 0.836 ± 0.006 | 0.692 ± 0.021    | 0.804 ± 0.011    |
| fpr               | 0.119 ± 0.011 | 0.046 ± 0.008    | 0.157 ± 0.030    |
| tpr_race          | 0.756 ± 0.029 | 0.490 ± 0.053    | 0.756 ± 0.043    |
| tpr_religion      | 0.705 ± 0.053 | 0.272 ± 0.056    | 0.625 ± 0.097    |
| tpr_origin        | 0.769 ± 0.040 | 0.318 ± 0.073    | 0.714 ± 0.069    |
| tpr_gender        | 0.827 ± 0.029 | 0.356 ± 0.095    | 0.798 ± 0.052    |
| tpr_sexuality     | 0.824 ± 0.032 | 0.624 ± 0.066    | 0.833 ± 0.038    |
| tpr_gap           | 0.151 ± 0.049 | 0.372 ± 0.081    | 0.222 ± 0.077    |
| label_quality     | 1.000 ± 0.000 | 0.744 ± 0.018    | 0.835 ± 0.022    |
| labels_spent      | —             | 870.600 ± 19.914 | 825.500 ± 33.786 |
| expert_spent      | —             | 0.000 ± 0.000    | 153.000 ± 0.000  |
| exposure_max      | —             | 113.300 ± 7.102  | 80.700 ± 11.221  |
| exposure_gini     | —             | 0.045 ± 0.016    | 0.111 ± 0.032    |

## Paired tests (area under the learning curve)

| metric (area under curve)   | comparison                | difference   | Wilcoxon p   |
|:----------------------------|:--------------------------|:-------------|:-------------|
| balanced_accuracy           | final system − naive HITL | +0.100       | < 0.0001     |
| balanced_accuracy           | final system − oracle     | -0.043       | < 0.0001     |
| fpr                         | final system − naive HITL | +0.145       | < 0.0001     |
| fpr                         | final system − oracle     | +0.066       | < 0.0001     |
| tpr_gap                     | final system − naive HITL | -0.118       | 0.0002       |
| tpr_gap                     | final system − oracle     | +0.066       | 0.0009       |
| worst_group_tpr             | final system − naive HITL | +0.360       | < 0.0001     |
| worst_group_tpr             | final system − oracle     | -0.074       | 0.0056       |
