"""Unit tests for the HRAgent query classification and routing logic.

These tests mock the RAG chain (ask) so they don't require an API key or ChromaDB.
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import (
    ESCALATION_MESSAGE,
    ESCALATION_PATTERNS,
    OFF_TOPIC_MESSAGE,
    OFF_TOPIC_PATTERNS,
    HRAgent,
)

# Reusable mock return value for the RAG chain
MOCK_RAG_RESULT = {
    "answer": "According to the vacation policy, you get 15 days PTO.",
    "sources": [{"source": "vacation_pto.md", "category": "Vacation & PTO", "page": None}],
    "has_context": True,
}


@pytest.fixture
def agent():
    return HRAgent()


class TestClassifyQuery:
    """Test _classify_query routing logic (no API calls needed)."""

    def test_hr_query_classification(self, agent):
        assert agent._classify_query("How many vacation days do I get?") == "HR_QUERY"
        assert agent._classify_query("What is the parental leave policy?") == "HR_QUERY"
        assert agent._classify_query("Tell me about dental benefits") == "HR_QUERY"

    @pytest.mark.parametrize("query", [
        "I'm being harassed at work",
        "My manager is discriminating against me",
        "I'm experiencing workplace bullying",
        "I want to report sexual harassment",
        "I'm having suicidal thoughts",
        "I'm being threatened by a coworker",
        "I need legal advice about my termination",
        "I was wrongfully dismissed",
        "I'm being retaliated against for reporting",
        "I want to be a whistleblower",
    ])
    def test_escalation_queries(self, agent, query):
        assert agent._classify_query(query) == "ESCALATE"

    @pytest.mark.parametrize("query", [
        "What's the weather today?",
        "Give me a recipe for pasta",
        "What's a good movie to watch?",
        "Write me some Python code",
        "Tell me a joke",
        "Who is the president of the USA?",
    ])
    def test_off_topic_queries(self, agent, query):
        assert agent._classify_query(query) == "OFF_TOPIC"


class TestProcess:
    """Test the full process() method with mocked RAG chain."""

    @patch("src.agent.ask", return_value=MOCK_RAG_RESULT)
    def test_hr_query_returns_rag_answer(self, mock_ask, agent):
        result = agent.process("How many vacation days do I have?")
        assert result["query_type"] == "HR_QUERY"
        assert result["has_context"] is True
        assert "vacation" in result["answer"].lower()
        assert len(result["sources"]) > 0

    @patch("src.agent.ask", return_value=MOCK_RAG_RESULT)
    def test_escalation_includes_warning(self, mock_ask, agent):
        result = agent.process("I'm being harassed at work")
        assert result["query_type"] == "ESCALATE"
        assert "sensitive matter" in result["answer"].lower()
        assert "HR" in result["answer"]

    def test_off_topic_response(self, agent):
        result = agent.process("Tell me a joke")
        assert result["query_type"] == "OFF_TOPIC"
        assert result["sources"] == []
        assert result["has_context"] is False
        assert "outside the scope" in result["answer"].lower()


class TestChatHistory:
    """Test conversation memory management."""

    @patch("src.agent.ask", return_value=MOCK_RAG_RESULT)
    def test_history_grows_after_process(self, mock_ask, agent):
        agent.process("What is the PTO policy?")
        assert len(agent.chat_history) == 2  # 1 HumanMessage + 1 AIMessage

    @patch("src.agent.ask", return_value=MOCK_RAG_RESULT)
    def test_history_trimmed_at_window(self, mock_ask, agent):
        """After exceeding MEMORY_WINDOW_SIZE turns, old messages are trimmed."""
        for i in range(10):
            agent.process(f"Question number {i}")
        # MEMORY_WINDOW_SIZE=5, so max 10 messages (5 turns × 2)
        assert len(agent.chat_history) <= 10

    def test_clear_history(self, agent):
        agent.chat_history = [HumanMessage(content="hi"), AIMessage(content="hello")]
        agent.clear_history()
        assert agent.chat_history == []
