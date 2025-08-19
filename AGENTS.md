# AgenticAI Multi-Agent System Documentation

## Overview

AgenticAI is a sophisticated multi-agent system built using **LangChain** and **LangGraph** that demonstrates autonomous AI workflows. The system orchestrates multiple specialized agents to handle different types of queries through intelligent routing, validation, and feedback loops.

### Key Features

- **Graph-based Architecture**: Uses LangGraph's StateGraph for complex decision-making workflows
- **Intelligent Routing**: Supervisor agent dynamically routes queries to appropriate specialized agents
- **Validation Feedback Loops**: Built-in validation ensures response quality with retry mechanisms
- **Modular Design**: Clean separation of concerns with dedicated modules for agents, models, routers, and utilities
- **Multi-domain Capabilities**: Handles general knowledge, document retrieval, stock data, and web search

## System Architecture

The system implements a **StateGraph** workflow with the following components:

```
User Query → Supervisor Agent → Router → Specialized Agent → Validation → Response
                    ↑                                            ↓
                    ←←←←←←←← Retry Loop (if validation fails) ←←←←←
```

### Core Components

1. **AgentState**: TypedDict managing message sequences throughout the workflow
2. **StateGraph**: LangGraph's graph-based execution engine
3. **Conditional Routing**: Dynamic agent selection based on query content
4. **Validation Layer**: Quality assurance with retry mechanisms
5. **Pydantic Parsers**: Structured output validation and parsing

## Agent Specifications

### 1. Supervisor Agent

**Purpose**: Central orchestrator that classifies and routes user queries to appropriate specialized agents.

**Capabilities**:
- Query classification using GPT-4o
- Pydantic-based structured output parsing
- Dynamic routing decisions

**Input Format**:
```python
{
    "messages": ["User query string"]
}
```

**Output Format**:
```python
{
    "messages": ["Agent_Type"]  # One of: "Avijit", "LLM", "Latest", "Stock Price"
}
```

**Classification Logic**:
- **"Avijit"** → Routes to RAG Agent for resume-related queries
- **"LLM"** → Routes to LLM Agent for general knowledge
- **"Latest"** → Routes to Web Agent for current information
- **"Stock Price"** → Routes to Stock Agent for financial data

**Implementation**: `agentic_ai/code/agentic_ai/nodes/supervisor.py`

### 2. LLM Agent

**Purpose**: Handles general knowledge queries using OpenAI's GPT-4o model.

**Capabilities**:
- General knowledge question answering
- Step-by-step instructions
- Explanations and educational content

**Input Format**:
```python
{
    "messages": ["How to make black tea?, list down in simple steps."]
}
```

**Output Format**:
```python
{
    "messages": ["Detailed step-by-step response with formatting"]
}
```

**Use Cases**:
- How-to guides and tutorials
- General knowledge questions
- Educational explanations
- Creative content generation

**Implementation**: `agentic_ai/code/agentic_ai/nodes/agents/llm_call.py`

### 3. RAG Agent (Retrieval-Augmented Generation)

**Purpose**: Answers questions about Avijit Bhattacharjee's resume using document retrieval and generation.

**Capabilities**:
- PDF document processing and chunking
- FAISS vector store for semantic search
- HuggingFace embeddings (all-MiniLM-L6-v2)
- Context-aware response generation

**Technical Details**:
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 100 characters
- **Embedding Model**: all-MiniLM-L6-v2
- **Vector Store**: FAISS (Facebook AI Similarity Search)

**Input Format**:
```python
{
    "messages": ["Who is Avijit Bhattacharjee? Does he have any cloud certifications?"]
}
```

**Output Format**:
```python
{
    "messages": ["Context-based response from resume data"]
}
```

**Use Cases**:
- Resume information queries
- Professional background questions
- Skills and certification inquiries
- Experience and project details

**Implementation**: `agentic_ai/code/agentic_ai/nodes/agents/rag.py`

### 4. Stock Agent

**Purpose**: Retrieves real-time stock market data using the yfinance API.

**Capabilities**:
- Real-time stock price retrieval
- Company name to ticker symbol mapping
- Price change calculations and analysis
- Multi-currency support

**Supported Companies**:
```python
{
    "nvidia": "NVDA", "apple": "AAPL", "microsoft": "MSFT",
    "google": "GOOGL", "amazon": "AMZN", "meta": "META", "tesla": "TSLA"
}
```

**Input Format**:
```python
{
    "messages": ["what is the current stock price of Nvidia?"]
}
```

**Output Format**:
```python
{
    "messages": ["The current stock price of NVIDIA Corporation (NVDA) is $XXX.XX USD, up $X.XX USD (X.XX%) from the previous close."]
}
```

**Data Points Provided**:
- Current stock price
- Previous close price
- Price change (absolute and percentage)
- Currency information
- Company full name

**Implementation**: `agentic_ai/code/agentic_ai/nodes/agents/stock_price.py`

### 5. Web Agent

**Purpose**: Performs real-time web searches using DuckDuckGo for current information.

**Capabilities**:
- Real-time web search
- Current events and news
- Latest information retrieval
- Privacy-focused search (DuckDuckGo)

**Input Format**:
```python
{
    "messages": ["Who is the CM of west bengal?"]
}
```

**Output Format**:
```python
{
    "messages": ["Search results with current information"]
}
```

**Use Cases**:
- Current events and news
- Real-time information queries
- Latest updates on topics
- Fact-checking current information

**Implementation**: `agentic_ai/code/agentic_ai/nodes/agents/search_func.py`

### 6. Validation Agent

**Purpose**: Ensures response relevance and quality with retry mechanisms.

**Capabilities**:
- Response relevance validation
- Quality assurance checks
- Retry loop coordination
- Structured validation output

**Validation Process**:
1. Compares user query with agent response
2. Determines relevance using GPT-4o
3. Provides structured feedback with reasoning
4. Triggers retry if validation fails

**Input Format**:
```python
{
    "messages": ["Original query", "Agent response"]
}
```

**Output Format**:
```python
{
    "messages": ["Yes" or "No"]  # Validation result
}
```

**Validation Criteria**:
- Response relevance to original query
- Accuracy of information provided
- Completeness of the answer
- Appropriate formatting and structure

**Implementation**: `agentic_ai/code/agentic_ai/nodes/agents/validation.py`

## Technical Architecture

### StateGraph Workflow

The system uses LangGraph's StateGraph to manage complex workflows:

```python
workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", supervisor)
workflow.add_node("RAG", custom_rag.rag)
workflow.add_node("LLM", llm_call)
workflow.add_node("WEB", search_func)
workflow.add_node("STOCK", stock_price_tool)
workflow.add_node("VALIDATION", validation)
```

### Conditional Routing

The router function dynamically selects agents based on query content:

```python
def router(state: AgentState):
    last_message = state["messages"][-1]
    if "avijit" in last_message.lower():
        return "RAG Call"
    elif any(word in last_message.lower() for word in ["stock price", "stock", "share price"]):
        return "STOCK Call"
    elif "latest" in last_message.lower():
        return "WEB Call"
    else:
        return "LLM Call"
```

### Validation Feedback Loop

The validation router determines whether to end the workflow or retry:

```python
workflow.add_conditional_edges("VALIDATION", validation_router_1, {"yes": END, "no": "Supervisor"})
```

## Configuration Requirements

### Environment Variables

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Model Specifications

**Primary LLM**: GPT-4o (OpenAI)
- Used for: Supervisor routing, LLM agent responses, validation
- Configuration: `gpt_model_name = "gpt-4o"`

**Embedding Model**: all-MiniLM-L6-v2 (HuggingFace)
- Used for: RAG agent document embeddings
- Configuration: `embedding_model_name = "all-MiniLM-L6-v2"`

### Python Requirements

- **Python Version**: 3.9+
- **Key Dependencies**: See Dependencies section below

## Usage Examples

### Example 1: General Knowledge Query (LLM Agent)

**Input** (`agentic_ai/data/input/llm_message.txt`):
```
How to make black tea?, list down in simple steps.
```

**Output** (`agentic_ai/data/output/llm_message.json`):
```json
{
  "messages": [
    "How to make black tea?, list down in simple steps.",
    "LLM",
    "Making black tea is a straightforward process. Here are the simple steps:\n\n1. **Boil Water**: Start by boiling fresh, filtered water...",
    "Yes"
  ]
}
```

### Example 2: Resume Query (RAG Agent)

**Input** (`agentic_ai/data/input/rag_message.txt`):
```
Please answer the following:
1. Who is Avijit Bhattacharjee?
2. Does he have any cloud certifications?
```

**Output** (`agentic_ai/data/output/rag_message.json`):
```json
{
  "messages": [
    "Please answer the following:\n1. Who is Avijit Bhattacharjee?\n2. Does he have any cloud certifications?",
    "Avijit",
    "Based on the resume information provided...",
    "Yes"
  ]
}
```

### Example 3: Stock Price Query (Stock Agent)

**Input** (`agentic_ai/data/input/stock_price_message.txt`):
```
what is the current stock price of Nvidia?
```

**Output** (`agentic_ai/data/output/stock_price_message.json`):
```json
{
  "messages": [
    "what is the current stock price of Nvidia?",
    "Stock Price",
    "The current stock price of NVIDIA Corporation (NVDA) is $XXX.XX USD...",
    "Yes"
  ]
}
```

### Example 4: Web Search Query (Web Agent)

**Input** (`agentic_ai/data/input/web_message.txt`):
```
Who is the CM of west bengal?
```

**Output** (`agentic_ai/data/output/web_message.json`):
```json
{
  "messages": [
    "Who is the CM of west bengal?",
    "Latest",
    "Current information about West Bengal Chief Minister...",
    "Yes"
  ]
}
```

## Code Structure

The codebase follows a modular architecture with clear separation of concerns:

```
agentic_ai/code/agentic_ai/
├── nodes/
│   ├── supervisor.py          # Central routing logic
│   └── agents/
│       ├── llm_call.py        # General knowledge agent
│       ├── rag.py             # Document retrieval agent
│       ├── stock_price.py     # Financial data agent
│       ├── search_func.py     # Web search agent
│       └── validation.py      # Response validation agent
├── models/
│   └── models.py              # LLM and embedding model configuration
├── routers/
│   └── router.py              # Query routing and validation routing
├── utils/
│   ├── agent_state.py         # State management TypedDict
│   └── utils.py               # Helper functions and utilities
└── validations/
    └── validation.py          # Pydantic parsers and validators
```

### Key Modules

#### `models/models.py`
Configures and initializes LLM and embedding models:
```python
@dataclass
class Models:
    gpt_model_name: str = "gpt-4o"
    embedding_model_name: str = "all-MiniLM-L6-v2"
```

#### `utils/agent_state.py`
Defines the state structure for the workflow:
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
```

#### `validations/validation.py`
Pydantic models for structured output parsing:
```python
class TopicSelectionParser(BaseModel):
    Topic: str = Field(description="selected topic")
    Reasoning: str = Field(description="Reasoning behind topic selection")
```

## Dependencies

### Core Dependencies

```txt
langchain>=0.3.27              # LLM orchestration and chains
langgraph>=0.5.4               # Graph-based multi-agent workflows
langchain-openai>=0.3.28       # OpenAI integration
langchain-community>=0.3.27    # Community tools and integrations
langchain-huggingface>=0.3.1   # HuggingFace model integration
```

### Specialized Dependencies

```txt
openai>=1.97.1                 # OpenAI API client
faiss-cpu>=1.11.0.post1        # Vector similarity search
sentence-transformers>=5.0.0   # Embedding models
yfinance>=0.2.65               # Stock market data
duckduckgo-search>=8.1.1       # Web search functionality
```

### Utility Dependencies

```txt
pydantic>=2.11.7               # Data validation and parsing
python-dotenv>=1.1.1           # Environment variable management
pypdf>=5.8.0                   # PDF document processing
transformers>=4.54.0           # Transformer model support
black>=25.1.0                  # Code formatting
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/avijit1996iiti/AgenticAI.git
cd AgenticAI
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

### 5. Run the System

```bash
python agentic_ai/code/main_supervisor_agentic_flow.py
```

## Running Examples

The main script includes commented examples for each agent type. Uncomment the desired example:

```python
# LLM Agent Example
run_agentic_app_from_file(
    app=app,
    input_path="agentic_ai/data/input/llm_message.txt",
    output_path="agentic_ai/data/output/llm_message.json"
)

# RAG Agent Example
run_agentic_app_from_file(
    app=app,
    input_path="agentic_ai/data/input/rag_message.txt",
    output_path="agentic_ai/data/output/rag_message.json"
)

# Stock Agent Example
run_agentic_app_from_file(
    app=app,
    input_path="agentic_ai/data/input/stock_price_message.txt",
    output_path="agentic_ai/data/output/stock_price_message.json"
)

# Web Agent Example
run_agentic_app_from_file(
    app=app,
    input_path="agentic_ai/data/input/web_message.txt",
    output_path="agentic_ai/data/output/web_message.json"
)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. OpenAI API Key Issues

**Problem**: `openai.AuthenticationError` or missing API key errors

**Solution**:
- Ensure `.env` file exists in project root
- Verify `OPENAI_API_KEY` is correctly set
- Check API key validity on OpenAI platform

```bash
# Check if .env file exists
ls -la .env

# Verify environment variable loading
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

#### 2. FAISS Installation Issues

**Problem**: FAISS-CPU installation failures on certain systems

**Solution**:
```bash
# For conda users
conda install -c conda-forge faiss-cpu

# For pip users with specific versions
pip install faiss-cpu==1.11.0.post1 --no-cache-dir
```

#### 3. HuggingFace Model Download Issues

**Problem**: Slow or failed embedding model downloads

**Solution**:
- Ensure stable internet connection
- Use HuggingFace cache directory:
```bash
export HF_HOME=/path/to/cache
```

#### 4. PDF Processing Errors

**Problem**: PyPDF errors when processing resume documents

**Solution**:
- Ensure PDF file exists at specified path
- Check PDF file integrity
- Verify file permissions

```python
# Debug PDF loading
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("agentic_ai/data/Avijit_Bhattacharjee__Senior_Machine_Learning_Engineer__L3__.pdf")
pages = loader.load()
print(f"Loaded {len(pages)} pages")
```

#### 5. Stock Data Retrieval Issues

**Problem**: yfinance API timeouts or invalid ticker symbols

**Solution**:
- Check internet connectivity
- Verify ticker symbol validity
- Add error handling for API failures

#### 6. Web Search Limitations

**Problem**: DuckDuckGo search rate limiting or blocked requests

**Solution**:
- Implement request delays
- Use alternative search backends
- Check network firewall settings

### Performance Optimization

#### 1. Vector Store Optimization

For large document collections:
```python
# Increase chunk size for better context
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increased from 500
    chunk_overlap=200  # Increased from 100
)
```

#### 2. Model Caching

Enable model caching to reduce initialization time:
```python
# Cache embedding models
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)
```

#### 3. Concurrent Processing

For multiple queries, consider async processing:
```python
import asyncio

async def process_multiple_queries(queries):
    tasks = [app.ainvoke({"messages": [query]}) for query in queries]
    return await asyncio.gather(*tasks)
```

## Extension Guidelines

### Adding New Agents

To add a new specialized agent to the system:

#### Step 1: Create Agent Implementation

Create a new file in `agentic_ai/code/agentic_ai/nodes/agents/`:

```python
# new_agent.py
from agentic_ai.utils.agent_state import AgentState

def new_agent(state: AgentState):
    print("-> New Agent ->")
    question = state["messages"][0]
    
    # Implement your agent logic here
    response = process_query(question)
    
    return {"messages": [response]}
```

#### Step 2: Update Supervisor Classification

Modify `agentic_ai/code/agentic_ai/nodes/supervisor.py`:

```python
template = """
Your task is to classify the given user query into one of the following categories: 
[Avijit, LLM, Latest, Stock Price, NewCategory]. 
# Add description for new category
"""
```

#### Step 3: Update Router Logic

Modify `agentic_ai/code/agentic_ai/routers/router.py`:

```python
def router(state: AgentState):
    last_message = state["messages"][-1]
    # Add new routing condition
    if "new_keyword" in last_message.lower():
        return "NEW_AGENT Call"
    # ... existing conditions
```

#### Step 4: Update Main Workflow

Modify `main_supervisor_agentic_flow.py`:

```python
from agentic_ai.nodes.agents.new_agent import new_agent

# Add node to workflow
workflow.add_node("NEW_AGENT", new_agent)

# Add conditional edge
workflow.add_conditional_edges(
    "Supervisor",
    router,
    {
        "RAG Call": "RAG",
        "LLM Call": "LLM",
        "WEB Call": "WEB",
        "STOCK Call": "STOCK",
        "NEW_AGENT Call": "NEW_AGENT",  # Add new route
    },
)

# Add validation edge
workflow.add_edge("NEW_AGENT", "VALIDATION")
```

### Best Practices for Extensions

#### 1. Error Handling

Implement robust error handling in new agents:

```python
def new_agent(state: AgentState):
    try:
        question = state["messages"][0]
        response = process_query(question)
        return {"messages": [response]}
    except Exception as e:
        error_message = f"Error in new agent: {str(e)}"
        return {"messages": [error_message]}
```

#### 2. Input Validation

Validate inputs before processing:

```python
def new_agent(state: AgentState):
    if not state.get("messages") or not state["messages"]:
        return {"messages": ["No input provided"]}
    
    question = state["messages"][0]
    if not isinstance(question, str) or not question.strip():
        return {"messages": ["Invalid input format"]}
    
    # Process valid input
    response = process_query(question)
    return {"messages": [response]}
```

#### 3. Logging and Monitoring

Add comprehensive logging:

```python
import logging

logger = logging.getLogger(__name__)

def new_agent(state: AgentState):
    logger.info(f"New agent processing query: {state['messages'][0]}")
    
    try:
        response = process_query(state["messages"][0])
        logger.info(f"New agent response generated successfully")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"New agent error: {str(e)}")
        raise
```

#### 4. Configuration Management

Use configuration files for agent settings:

```python
# config.py
NEW_AGENT_CONFIG = {
    "timeout": 30,
    "max_retries": 3,
    "api_endpoint": "https://api.example.com"
}

# new_agent.py
from .config import NEW_AGENT_CONFIG

def new_agent(state: AgentState):
    config = NEW_AGENT_CONFIG
    # Use configuration in agent logic
```

### Testing New Agents

Create test cases for new agents:

```python
# test_new_agent.py
import pytest
from agentic_ai.nodes.agents.new_agent import new_agent

def test_new_agent_basic():
    state = {"messages": ["test query"]}
    result = new_agent(state)
    assert "messages" in result
    assert len(result["messages"]) > 0

def test_new_agent_error_handling():
    state = {"messages": []}
    result = new_agent(state)
    assert "No input provided" in result["messages"][0]
```

Run tests with:
```bash
pytest test_new_agent.py -v
```

## Development Guidelines

### Code Standards

- **Python Version**: 3.9+
- **Formatting**: Use `black` for code formatting
- **Import Sorting**: Use `isort` for import organization
- **Linting**: Maintain `pylint` score ≥ 8.5
- **Documentation**: Document all agents with clear input/output specifications

### Testing Standards

- **Unit Tests**: Use `pytest` for unit testing
- **End-to-End Tests**: Run `pytest -m e2e` for workflow validation
- **Coverage**: Target ≥ 80% test coverage using `pytest-cov`
- **Test Files**: Follow `test_*.py` naming convention

### Pull Request Guidelines

**PR Title Format**:
- `"agent: description"` (for new agent creation)
- `"fix: description"` (for bug fixes)
- `"chore: description"` (for maintenance tasks)

**PR Requirements**:
- Short summary of changes
- Steps to reproduce (if applicable)
- Linked issue(s) or task reference
- Test coverage for new features
- Documentation updates

This comprehensive documentation provides a complete guide to understanding, using, and extending the AgenticAI multi-agent system. For additional support or questions, please refer to the project repository or create an issue for community assistance.

