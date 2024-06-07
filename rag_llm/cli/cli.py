from .utils import *
from rag_llm.utils.const import *
from rag_llm.prompt import Prompt, Memory

memory = Memory()
# avoid the tokenizers parallelism issue
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model = load_model('llama')

def generate(text, platform, rag, precalculated_rag=None):
    """
    This function will call and generate the commands from LLM
    Args:
        text: the text to be converted into a command.
        print_cmd: if True, only print the generated command.
    """
    text = " ".join(text)
    prompt = Prompt(memory)
    # load the LLM model
    # model = load_model(platform)

    response = model.to_command(prompt.gen_commands(text, rag, precalculated_rag), text)
    return response

def get_rag(query):
    return memory.query([query])

def save_rag(queries):
    id = 0
    for query in queries:
        print(query)
        memory.add_query(queries=[query])
        id += 1
        print(f"{id} - Saved")

def delete_rag():
    memory.delete()

def load_rag():
    commands = memory.get()
    if commands:
        metadatas = commands['metadatas']
        documents = commands['documents']
        idx = commands['ids']

        for i in range(len(idx)):
            print(f"""
                User Input: {documents[i]}
                Generated Commands: {metadatas[i]['response']}
                """)
    print(memory.count())

def remove_rag(ids):
    memory.delete_query(ids=ids)