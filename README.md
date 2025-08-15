
# AgenticAI

**Autonomous Multi-Agent AI Workflows with LangChain & LangGraph**

AgenticAI is a demonstration project showing how to build agentic AI workflows — systems where one or more AI agents work autonomously over long tasks, making decisions, using tools, and even collaborating with other AI agents to achieve a goal. This repository supports a Knowledge Sharing Session (KSS) presentation on the differences between Generative AI, AI Agents, and Agentic AI, followed by a practical demo of an agentic AI workflow.

## Introduction
This project was presented in a KSS session to:
1. Define AI Agents
2. Compare GenAI, AI Agents, and Agentic AI
3. Explain when to use each approach
4. Demonstrate a working Agentic AI system using LangChain and LangGraph

In short:  
**AI Agent = LLM + Tools**  
**Agentic AI** = Multiple AI Agents working together over long-running, decision-heavy tasks.

## Concepts Covered

### Generative AI (GenAI)
- Produces text, images, or other content.
- Best for simple content generation.

### AI Agents
- Think and act autonomously using LLMs and tools.
- Best for simple autonomous tasks with minimal decision-making.

### Agentic AI
- Multi-agent systems with decisions, loops, conditionals, and state tracking.
- Best for complex, multi-step reasoning and decision-making.

## LangChain vs LangGraph

| Feature          | LangChain | LangGraph |
|------------------|-----------|-----------|
| Workflow Type    | Linear    | Graph-based |
| Best For         | Simple pipelines (e.g., Q&A chatbot) | Complex decision-making workflows |
| Supports Loops & Conditional Routing | ❌ | ✅ |
| State Tracking   | Limited   | Built-in |

## Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation (RAG) helps LLMs answer questions using private/internal documents rather than relying on public training data (which risks hallucination). It requires vector databases (like Pinecone, Weaviate, FAISS) to store and search embeddings. The choice depends on data size and infrastructure.

## Architecture
A Supervisor Agent intelligently routes user queries to specialized agents and validates their responses.

**Specialized Agents include:**
- LLM Agent – General knowledge tasks.
- RAG Agent – Answers from private documents.
- Stock Agent – Retrieves market/financial data.
- Web Agent – Pulls real-time online information.

**Workflow Steps:**
1. User query received.
2. Supervisor Agent decides which sub-agent(s) to call.
3. Sub-agent executes task.
4. Validation step ensures response quality.
5. Final response returned to user.

## Demo Workflow
The provided example:
- Uses LangChain for LLM integration and tool binding.
- Uses LangGraph for routing logic and stateful execution.
- Demonstrates multi-agent orchestration with decision-making.

## Tech Stack
- LangChain – LLM pipelines
- LangGraph – Graph-based multi-agent orchestration
- OpenAI / Llama-based LLMs – Reasoning and content generation
- Vector DB – RAG capabilities
- Python 3.11+

## Installation
```bash
git clone https://github.com/avijit1996iiti/AgenticAI.git
cd AgenticAI
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Usage

```bash
python agentic_ai/code/main_supervisor_agentic_flow.py
```

This launches the multi-agent system and processes queries via the Supervisor Agent.

## Takeaways

* GenAI is great for content creation.
* AI Agents shine for simple autonomous reasoning and tool use.
* Agentic AI is the future for complex, stateful, multi-step problem solving.

## Acknowledgements

* Rahul Grover (DOE, Spring Financial) – for initiating KSS and giving me the platform to share this.
* Team – for your valuable time and engagement during the session.


