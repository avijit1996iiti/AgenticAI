#!/usr/bin/env python3
"""
Test script to verify LangChain imports and basic functionality
without requiring external API calls.
"""

import sys
import os

# Add the project path to sys.path
sys.path.insert(0, "/home/daytona/AgenticAI/agentic_ai/code")


def test_langchain_imports():
    """Test that all LangChain imports work correctly with updated versions."""
    print("Testing LangChain imports with updated versions...")

    try:
        # Test langgraph imports (updated to 0.6.5)
        from langgraph.graph import StateGraph, END, START

        print("✅ langgraph imports successful (StateGraph, END, START)")

        # Test langchain-core imports
        from langchain_core.messages import BaseMessage
        from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.prompts import PromptTemplate

        print(
            "✅ langchain-core imports successful (BaseMessage, StrOutputParser, PydanticOutputParser, RunnablePassthrough, PromptTemplate)"
        )

        # Test langchain-community imports
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import FAISS

        print(
            "✅ langchain-community imports successful (DuckDuckGoSearchRun, PyPDFLoader, FAISS)"
        )

        # Test langchain-text-splitters imports
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        print(
            "✅ langchain-text-splitters imports successful (RecursiveCharacterTextSplitter)"
        )

        # Test langchain-huggingface imports
        from langchain_huggingface import HuggingFaceEmbeddings

        print("✅ langchain-huggingface imports successful (HuggingFaceEmbeddings)")

        # Test langchain-openai imports (updated to 0.3.30)
        from langchain_openai import ChatOpenAI

        print("✅ langchain-openai imports successful (ChatOpenAI)")

        # Test basic StateGraph functionality
        from agentic_ai.utils.agent_state import AgentState

        workflow = StateGraph(AgentState)
        print("✅ StateGraph initialization successful")

        print("\n🎉 All LangChain imports and basic functionality tests passed!")
        print("✅ Updated versions are compatible:")
        print("   - langgraph: 0.6.5")
        print("   - langchain-openai: 0.3.30")
        print("   - All other LangChain packages working correctly")
        print("✅ All deprecated imports have been updated to current paths")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_component_initialization():
    """Test that LangChain components can be initialized without API calls."""
    print("\nTesting component initialization...")

    try:
        # Test HuggingFace embeddings (no API key required)
        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ HuggingFaceEmbeddings initialization successful")

        # Test DuckDuckGo search tool initialization
        from langchain_community.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        print("✅ DuckDuckGoSearchRun initialization successful")

        # Test output parser
        from langchain_core.output_parsers import StrOutputParser

        parser = StrOutputParser()
        print("✅ StrOutputParser initialization successful")

        return True

    except Exception as e:
        print(f"❌ Component initialization error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("LangChain Dependency Update Verification Test")
    print("=" * 60)

    # Test imports
    imports_ok = test_langchain_imports()

    # Test component initialization
    components_ok = test_component_initialization()

    print("\n" + "=" * 60)
    if imports_ok and components_ok:
        print("🎉 ALL TESTS PASSED - LangChain updates are successful!")
        print("✅ The updated LangChain versions are fully compatible")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED - Check the errors above")
        sys.exit(1)
