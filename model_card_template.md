# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model predicts if a salary is <=50k or greater than 50k.

## Intended Use

Group customers into segments to have a more targeted marketing operation.

## Training Data

Details about the training data can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).

Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))

Prediction task is to determine whether a person makes over 50K a year.

## Evaluation Data

Random sample of 20% of the training data. A random state has been fixed to replicate the results.

## Metrics

Example of metrics and expected accuracies.

| Metric     | Value              |
|------------|--------------------|
| Precision: | 0.7745241581259151 |
| Recall:    | 0.6600124766063631 |
| F1:        | 0.7126978780734253 |
| AUC:       | 0.798609704154354  |

## Ethical Considerations

The predicted salary group leads to discrimination downstream.
Therefore, any action derived from the salary group predictions should be evaluated in this light.

The next section provides an overview about the model's fairness and biases.

## Caveats and Recommendations

### Metrics

**Selection rate**: Share of labels predicted positive (1)

**Demographic parity difference**: Difference between the largest and the smallest group-level selection rate, where a difference of 0 means that all groups have the same selection rate.

**Demographic parity ratio**: Difference between the largest and the smallest group-level selection rate, where 1 means that all groups have the same selection rate.

**Overall balanced error rate**: [Sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score)

Refer to other metrics in fairlearns [docs](https://fairlearn.org/v0.7.0/user_guide/assessment.html?highlight=balanced_accuracy_score).

### Fairness metrics

|                                | sex         | race        |
|--------------------------------|-------------|-------------|
| Overall selection rate         | 0.208666257 | 0.208666257 |
| Demographic parity difference  | 0.197307569 | 0.193720565 |
| Demographic parity ratio       | 0.275169865 | 0.137062937 |
| Overall balanced error rate    | 0.19602243  | 0.19602243  |
| Balanced error rate difference | 0.012387311 | 0.383196721 |
| False positive rate difference | 0.069625601 | 0.064382896 |
| False negative rate difference | 0.094400223 | 0.75        |
| Equalized odds difference      | 0.094400223 | 0.75        |
| Overall AUC                    | 0.80397757  | 0.80397757  |
| AUC difference                 | 0.012387311 | 0.383196721 |

### Fairness assessment

The model performs almost equally well with regard to gender.

However, there is a strong bias for race. A high share of false negatives drives this imbalance in model performance.

### Recommendations

There are several approaches to increase fairness:

* make the model agnostic about race and drop it as a feature
* use the `Gridsearch` mitigation method of Fairlearn: https://fairlearn.org/v0.7.0/api_reference/fairlearn.reductions.html#fairlearn.reductions.GridSearch
* refer to [`Mitigation`](https://fairlearn.org/v0.7.0/user_guide/mitigation.html) for further methods that increase fairness
