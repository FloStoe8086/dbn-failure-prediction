# DNB Failure Prediction

## Project Structure

```
.
├── README.md
└── src
    ├── alarm_features.py
    ├── bn_pgmpy.py
    ├── dbn_pgmpy.py
    ├── generate_transitions.py
    ├── stats.py
    └── xlsx_to_csv.py
```

## Approach 
Count and Duration features per alarm ID were extracted and discretized for each time window. The machine state was labeled as _Failure_ if at least one alarm occured during a failure state within that window. Consecutive window pairs (t -> t+1) were generated to create transitions.

A _DiscreteBayesianNetwork_ (pgmpy) was trained using DiscreteMLE (by default) on 80% of the transitions, with _next\_machine\_state_ state as prediction target. For the _DynamicBayesianNetwork_ approach see **Limitations**.

## Results
The model achieved 86% accuracy on the test set. Failure states were predicted with 90% precision and 88% recall. Although there was a sparse data problem, the most frequent alarm patterns were sufficient to distinguish failure from running.

## Limitations
A _DynamicBayesianNetwork_ was attempted where the model was successfully trained (CPTs estimated via MLE) but inference could not be performed. _DBNInference_ resulted in a `ValueError: Node not present in the model` during inference initialization, which could not be solved and therefore prevented evaluation. Therefore, _DiscreteBayesianNetwork_ was used instead. 