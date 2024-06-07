import transformers
import torch
from rag_llm.prompt import extract_shell_commands

class Llama3Model:
  def __init__(self):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    self.pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda:2",
    )
    self.terminators = [
        self.pipeline.tokenizer.eos_token_id,
        self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
  
  def to_command(self, prompt, text):
    input = f"""
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{text}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
    """
    outputs = self.pipeline(
      input,
      max_new_tokens=256,
      eos_token_id=self.terminators,
      do_sample=False,
      temperature=None,
      top_p=None,
      pad_token_id=self.pipeline.tokenizer.eos_token_id
    )
    print(outputs[0]["generated_text"][len(input):])
    return {"cmd" : extract_shell_commands(outputs[0]["generated_text"][len(input):]), "confidence" : 100}