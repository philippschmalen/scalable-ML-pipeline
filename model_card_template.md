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

If the predicted salary group leads to discrimintation downstream.
Therefore, any action derived from the salary group predictions should be evaluated in this light.

## Caveats and Recommendations
