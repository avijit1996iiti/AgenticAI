<general_rules>

- Always use Python 3.9+ for agent development  
- Follow PEP8 coding standards and use `black` + `isort` for formatting  
- Use `pylint` for linting and maintain a minimum score of 8.5  
- Document all agents with clear input/output specifications  

</general_rules>

<repository_structure>
This repository contains modular agent implementations for the Agentic AI demo:

- agentic_ai/code/agentic_ai/nodes: Creates all nodes of the graph
- agentic_ai/code/agentic_ai/nodes/agents: Has the required code to create all agents
- agentic_ai/code/agentic_ai/models: Has code base rerquired to use different LLMs
- agentic_ai/code/agentic_ai/routers: Has code base rerquired to use different routers in the Agentic AI project   
- agentic_ai/code/agentic_ai/utils: Common helper functions and constants
- agentic_ai/code/agentic_ai/validations: Code for validating LLMs output using pydantic

 

</repository_structure>

<dependencies_and_installation>

Run `pip install -r requirements.txt` from the repository root to install all dependencies.  
Key dependencies include:  

- `langchain` for agent orchestration  
- `openai` for LLM integration  
- `pytest` for testing  
- `boto3` for AWS integrations (optional for deployment workflows)  

</dependencies_and_installation>

<testing_instructions>

- Run `pytest` for unit tests  
- Run `pytest -m e2e` for end-to-end agent flow validation  
- Test files follow the `test_*.py` naming convention  
- Use `pytest-cov` to measure test coverage (target ≥ 80%)  

</testing_instructions>

<pull_request_formatting>

PR titles should follow:  
- `"agent: description"` (for new agent creation)  
- `"fix: description"` (for bug fixes)  
- `"chore: description"` (for maintenance tasks)  

Each PR must include:  
- A short summary of changes  
- Steps to reproduce (if applicable)  
- Linked issue(s) or task reference  

</pull_request_formatting>
