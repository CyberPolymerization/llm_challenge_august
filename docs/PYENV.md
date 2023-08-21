

This document outlines steps to set and install the challenge's python environment and other dependencies.

1. [One-Time/Installation Instructions](#onetime)
- [Prerequisites](#preq)
- [Poetry Installation](#poet)
- [OpenAI API Key](#api)
2. [All-Time Instructions](#dev)

---
### One-Time/Installation Instructions <a name="onetime"></a>

#### Prerequisites <a name="preq"></a>
Before we begin, let's make sure we have the following prerequisites installed:

- Python (version 3.6 or higher)
- pip (Python package installer)

More information on the above can be found [here](https://www.python.org/downloads/).

Next, we will walk through the installation process of Poetry, a powerful Python package manager and dependency solver to install all libraries we need for the challenge.

#### Poetry Installation <a name="poet"></a>

To install Poetry, follow these steps:

1. Open your terminal or command prompt.

2. Run the following command to install Poetry using pip:

   ```shell
   pip install poetry
   ```
   If you encounter permission errors, you may need to run the command with administrative privileges or use a virtual environment.

3. Once the installation is complete, verify that Poetry is installed correctly by running the following command:

   ```shell
   poetry --version
   ```

   You should see the version number of Poetry printed in the terminal.

More information on managing dependencies and other Poetry commands can be found in the [official documentation](https://python-poetry.org/docs/).

4. Install our LLM Challenge package and its dependencies from within this repository main directory (after downloading/cloning this repository)

   ```shell
   cd llm_challenge_as23
   poetry install
   ```

#### OpenAI API Key <a name="api"></a>

In this tutorial/challenge, we make use of several Large Language Model (LLM)-based endpoints offered by OpenAI to interact with well-trained LLMs. That said, you are free to use other LLMs (e.g., open-source) if you like.
If you would like to interact with OpenAI's LLMs, you will be able to do so for free with $5 in free credit that can be used during your first 3 months. You need to create an OpenAI account [here](https://platform.openai.com/login?launch). We are cautiously optimistic that this is enough for participants given that we provide you with pre-computed data so as not to re-run yourself through an LLM. We recommend running your experiements on small subset of Q&As and run through the full dataset only when you would like to make a submission to make a good use of this free credit.

After creating an OpenAI account, you can find your secret API key [here](https://platform.openai.com/account/api-keys) .
Do not share your API key with others, or expose it in the browser or other client-side code. 
It is recommended that you put your OpenAI API key in an `.env` file as follows.

   ```shell
   echo OPENAI_API_KEY=$YOUR_OPENAI_API_KEY > .env
   ```

Later on, in the code (be it a script or Jupyter notebook), the API key will be fetched automatically with the following helper Python function
   ```python
   from llm_challenge.utils.misc import set_openai_api_key
   set_openai_api_key()
   ```

Note: If you run out of free OpenAI credits, we encourage you to play with [LLAMA 2](https://ai.meta.com/resources/models-and-libraries/llama/).


---
### All-Time Instructions <a name="dev"></a>

The above steps are needed to be done once and the outcome is a Python environment, named something like `llm_challenge-py3.8` . To run scripts/notebooks from a terminal, we need to activate this environment in the terminal as follows.
```shell
poetry shell
```

