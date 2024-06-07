import os
import platform
import subprocess

from rag_llm.prompt import Memory
from rag_llm.agent import OpenAIModel, ClaudeModel, Llama3Model
from rag_llm.utils.const import *


def load_model(platform: str = CONFIG_SEC_OPENAI):
    """
    load_model: load the model based on the configuration.
    """
    if platform == CONFIG_SEC_OPENAI:
        model = OpenAIModel(
            api_key="", version="gpt-3.5-turbo",
            temperature=0
        )
    elif platform == CONFIG_SEC_CLAUDE:
        model = ClaudeModel(
            api_key="", version="",
            generation_config={
                'stop_sequences': None,
                'temperature': 0,
                'top_p': 1.0,
                'top_k': 32,
                'max_tokens': 1500
            }
        )
    elif platform == CONFIG_SEC_LLAMA:
        model = Llama3Model()
    else:
        raise ValueError(f"Platform {platform} not supported.")
    return model


def execute_command(command: str) -> bool:
    """
    Execute a command and return whether it was successful.

    Args:
        command: The command to execute.

    Returns:
        True if the command succeeded, False otherwise.
    """
    try:
        if platform.system() == "Windows":
            is_powershell = len(os.getenv("PSModulePath", "").split(os.pathsep)) >= 3
            if is_powershell:
                # Powershell execution
                completed = subprocess.run(['powershell.exe', '-Command', command], check=True)
            else:
                # CMD execution
                completed = subprocess.run(['cmd.exe', '/c', command], check=True)
        else:
            # Unix-like shell execution
            shell = os.environ.get("SHELL", "/bin/sh")
            completed = subprocess.run([shell, '-c', command], check=True)

        return completed.returncode == 0
    except subprocess.CalledProcessError:
        # The command failed
        return False


def save_command(command: str, text: str, memory: Memory):
    """
    save_command: save the command into database.
    Args:
        command: the command to execute.
        text: the user prompt.
        config_dict: config dictionary
        memory: vector database in memory
    """
    # add the query to the memory, eviction with the default max size of 2000.
    storage_size = 2000

    if memory.count() > storage_size:
        memory.delete()

    if command != '':
        memory.add_query(queries=[{"query": text, "response": command}])
