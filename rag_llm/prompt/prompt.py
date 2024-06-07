from .memory import Memory
from rag_llm.utils.metadata import *
from rag_llm.utils import CONFIG_SEC_OPENAI

import textwrap
from datetime import datetime


class Prompt:
    def __init__(self, memory):
        """
        Prompt for Termax: the prompt for the LLMs.
        Args:
            memory: the memory instance.
        """
        # TODO: make the sync of system related metadata once happened at the initialization
        self.system_metadata = get_system_metadata()
        self.path_metadata = get_path_metadata()
        self.command_history = get_command_history()

        # share the same memory instance.
        if memory is None:
            self.memory = Memory()
        else:
            self.memory = memory


    def gen_commands(self, text: str, rag: bool = True, precalculated_rag: str = None):
        """
        [Prompt] Convert the natural language text to the commands.
        Args:
            text: the natural language text.
            model: the model to use, default is OpenAI.
        """
        if not (precalculated_rag or rag):
            # query the history database to get similar samples
            samples = self.memory.query([text])
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
        else:
            sample_string = precalculated_rag

        # refresh the metadata
        files = get_file_metadata()
        if rag:
            return textwrap.dedent(
                f"""\
                You are an shell expert, you can convert natural language text from user to shell commands.
                
                1. Please provide only shell commands for os without any description.
                2. Ensure the output is a valid shell command.
                3. If multiple steps required try to combine them together.
                
                Here are some rules you need to follow:

                1. The commands should be able to run on the current system according to the system information.
                2. The files in the commands should be available in the path, according to the path information.
                3. The CLI application should be installed in the system (check the path information).

                Here are some information you may need to know:
                
                [INFORMATION] The user's current system information:
                1. OS: {self.system_metadata['platform']}
                2. OS Version: {self.system_metadata['platform_version']}
                3. Architecture: {self.system_metadata['architecture']}
                
                [INFORMATION] The user's current PATH information:
                1. User: {self.path_metadata['user']}
                2. Current PATH: {self.path_metadata['current_directory']}
                3. Files under the current directory: {files['files']}
                4. Directories under the current directory: {files['directory']}
                5. Invisible files under the current directory: {files['invisible_files']}
                6. Invisible directories under the current directory: {files['invisible_directory']}
    
                Here are some similar commands generated before:
                {sample_string}

                The output shell commands is (please replace the `{{commands}}` with the actual commands):

                Commands: ${{commands}}
                """
            )
        else:
            return textwrap.dedent(
                f"""\
                You are an shell expert, you can convert natural language text from user to shell commands.
                
                1. Please provide only shell commands for os without any description.
                2. Ensure the output is a valid shell command.
                3. If multiple steps required try to combine them together.
                
                Here are some rules you need to follow:

                1. The commands should be able to run on the current system according to the system information.
                2. The files in the commands should be available in the path, according to the path information.
                3. The CLI application should be installed in the system (check the path information).

                The output shell commands is (please replace the `{{commands}}` with the actual commands):

                Commands: ${{commands}}
                """
            )