from rag_llm import generate, load_rag, delete_rag, save_rag, get_rag, remove_rag
from metric import  metric_utils
import json
import argparse

def load_data_in_rag():
    data = []
    with open("nl2bash-data_remaining.json", "r") as f:
        nl2bash = json.load(f)
    for id, sample in nl2bash.items():
        data.append({
            "query": sample["invocation"],
            "response": sample["cmd"]
        })

    with open("nl2cmd_data.json", "r") as f:
        nl2cmd = json.load(f)
    for id, sample in nl2cmd.items():
        data.append({
            "query": sample["invocation"],
            "response": sample["cmd"]
        })
    save_rag(data)
    return data

def pre_calculate_rag():
    with open("test.json", "r") as f:
        data = json.load(f)
    
    ids = []
    for id, sample in data.items():
        query = sample["invocation"]
        samples = get_rag(query)
        if samples['distances'][0][0] == 0.0:
            ids.append(samples['ids'][0][0])
    if ids:
        remove_rag(ids)
    
    for id, sample in data.items():
        query = sample["invocation"]
        samples = get_rag(query)
        metadatas = samples['metadatas'][0]
        documents = samples['documents'][0]
        distances = samples['distances'][0]

        # construct a string that contains the samples in a human-readable format
        sample_string = ""
        for i in range(len(documents)):
            sample_string += f"""
            User Input: {documents[i]}
            Generated Commands: {metadatas[i]['response']}
            Distance Score: {distances[i]}\n
            """
        data[id]["rag"] = sample_string
    with open("test_rag.json", "w") as f:
        json.dump(data, f)      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default='openai', choices=['openai', 'claude', 'llama'])
    args = parser.parse_args()
    print(args.model)
    load_data_in_rag()
    pre_calculate_rag()
    
    with open("test_rag.json", "r") as f:
        data = json.load(f)
    print(data)
    
    for id, sample in data.items():
        invocation = sample["invocation"]
        groundtruth = sample["cmd"]
        precalculated_rag = sample["rag"]
        
        response = generate(text=invocation, platform="llama", rag=False, precalculated_rag=precalculated_rag)
        metric_val = metric_utils.compute_metric(response['cmd'], response['confidence'], groundtruth)
        response.update({'score': metric_val})
        print(response)
        sample["baseline_llama"] = response
        
        response = generate(text=invocation, platform="llama", rag=True, precalculated_rag=precalculated_rag)
        metric_val = metric_utils.compute_metric(response['cmd'], response['confidence'], groundtruth)
        response.update({'score': metric_val})
        print(response)
        sample["rag_llama"] = response
    
    with open("test_rag.json", "w") as f:
        json.dump(data, f)
    
    rag = 0
    baseline = 0
    for id, sample in data.items():
        rag += sample["rag_llama"]["score"]
        baseline += sample["baseline_llama"]["score"]
    
    print("RAG", rag/len(data))
    print("Baseline", baseline/len(data))
