Metrics: bashlint and metric are our evaluation metric function.
Model: rag_llm contains all support for model and RAG system.
Experiment: main.py uses function and model from rag_llm and metric to perform experiment.

To run the experiment, simply do 
```
python3 main.py
```
To specify the model (claude, openai, llama), do
```
python3 main.py --model MODEL_TO_USE
```
To use claude, openai, specify api key in rag_llm/cli/utils.py in the load_model function

Key functions to get model is under rag/llm/cli.py file.
Key functions to get evaluation metric is from metric import metric_utils, and call
```
etric_val = metric_utils.compute_metric(response['cmd'], response['confidence'], groundtruth)
```

The test_rag.json files are dataset and experiment results.
