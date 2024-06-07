Metrics: bashlint and metric are our evaluation metric function.
Model: rag_llm contains all support for model and RAG system.
Experiment: main.py uses function and model from rag_llm and metric to perform experiment.

To run the experiment, simply do ```python3 main.py```.
Key functions to get model is under rag/llm/cli.py file.
Key functions to get evaluation metric is from metric import metric_utils, and call ```etric_val = metric_utils.compute_metric(response['cmd'], response['confidence'], groundtruth)```

The test_rag.json files are dataset and experiment results.
