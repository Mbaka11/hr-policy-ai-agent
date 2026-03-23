"""Agent layer — query classification, routing, and escalation logic."""

import logging
import re

from langchain_core.messages import AIMessage, HumanMessage

from src.chain import ask
from src.config import MEMORY_WINDOW_SIZE

logger = logging.getLogger(__name__)

# Keywords/patterns that trigger escalation to HR
ESCALATION_PATTERNS = [
    r"\bharass(ment|ed|ing)?\b",
    r"\bdiscriminat(ion|ed|ing|ory)?\b",
    r"\bbully(ing|ied)?\b",
    r"\bsexual\s+(assault|harassment|misconduct)\b",
    r"\bsuicid(e|al)\b",
    r"\bself[- ]harm\b",
    r"\bthreat(en|ened|ening|s)?\b",
    r"\bviolence\b",
    r"\bwhistleblow(er|ing)?\b",
    r"\bretali(at(ion|ed|ing)|atory)\b",
    r"\blawsuit\b",
    r"\blegal\s+(action|advice|rights|dispute)\b",
    r"\bterminate(d)?\b",
    r"\bwrongful(ly)?\s+(dismiss(al|ed)|terminat(ion|ed))\b",
    r"\bfir(ed|ing)\s+me\b",
]

ESCALATION_MESSAGE = (
    "⚠️ **This sounds like a sensitive matter that requires direct human support.**\n\n"
    "I understand this is important to you. While I can share general policy information, "
    "this type of situation requires personalized guidance from the HR team.\n\n"
    "**Please contact HR directly:**\n"
    "- Reach out to your HR representative\n"
    "- For urgent matters involving safety, contact your manager or security immediately\n"
    "- For confidential support, contact the Employee & Family Assistance Program (EFAP)\n\n"
)

# Patterns for clearly off-topic questions
OFF_TOPIC_PATTERNS = [
    r"\b(weather|recipe|movie|sport|game|stock|crypto|bitcoin)\b",
    r"\b(code|program|debug|javascript|python|sql|html)\b",
    r"\bwho\s+(is|was)\s+(the\s+)?(president|prime minister|king|queen)\b",
    r"\b(joke|funny|sing|poem|story)\b",
]

OFF_TOPIC_MESSAGE = (
    "That question falls outside the scope of HR policy information I can assist with. "
    "I'm here to help with questions about company policies such as:\n\n"
    "- 🏖️ Vacation & leave policies\n"
    "- 💊 Employee benefits (health, dental, insurance)\n"
    "- 📚 Training & development programs\n"
    "- 📊 Performance evaluation processes\n"
    "- 📋 Code of conduct & workplace standards\n\n"
    "Is there an HR policy question I can help you with?"
)


class HRAgent:
    """HR Policy AI Agent with query classification, routing, and memory.

    Classifies incoming queries as:
    - ESCALATE: Sensitive topics requiring human HR intervention
    - OFF_TOPIC: Questions unrelated to HR policies
    - HR_QUERY: Legitimate HR policy questions → routed to RAG chain
    """

    def __init__(self) -> None:
        self.chat_history: list[HumanMessage | AIMessage] = []

    def _classify_query(self, query: str) -> str:
        """Classify a user query into one of: ESCALATE, OFF_TOPIC, HR_QUERY."""
        query_lower = query.lower()

        # Check escalation patterns
        for pattern in ESCALATION_PATTERNS:
            if re.search(pattern, query_lower):
                logger.info("Query classified as ESCALATE: '%s'", query[:60])
                return "ESCALATE"

        # Check off-topic patterns
        for pattern in OFF_TOPIC_PATTERNS:
            if re.search(pattern, query_lower):
                logger.info("Query classified as OFF_TOPIC: '%s'", query[:60])
                return "OFF_TOPIC"

        return "HR_QUERY"

    def _trim_history(self) -> None:
        """Keep only the last N turns of conversation history."""
        max_messages = MEMORY_WINDOW_SIZE * 2  # Each turn = 1 human + 1 AI message
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]

    def process(self, query: str) -> dict:
        """Process a user query through classification and routing.

        Args:
            query: The user's message.

        Returns:
            Dictionary with keys:
            - answer: The response text
            - sources: List of source metadata (empty for non-RAG responses)
            - query_type: ESCALATE | OFF_TOPIC | HR_QUERY
            - has_context: Whether retrieved docs were found (only for HR_QUERY)
        """
        query_type = self._classify_query(query)

        if query_type == "ESCALATE":
            # Still retrieve relevant policy info to include after escalation message
            rag_result = ask(query, chat_history=self.chat_history)
            answer = ESCALATION_MESSAGE
            if rag_result["has_context"]:
                answer += (
                    "In the meantime, here's some general policy information that may be relevant:\n\n"
                    + rag_result["answer"]
                )
            result = {
                "answer": answer,
                "sources": rag_result.get("sources", []),
                "query_type": query_type,
                "has_context": rag_result["has_context"],
            }

        elif query_type == "OFF_TOPIC":
            result = {
                "answer": OFF_TOPIC_MESSAGE,
                "sources": [],
                "query_type": query_type,
                "has_context": False,
            }

        else:  # HR_QUERY
            rag_result = ask(query, chat_history=self.chat_history)
            result = {
                "answer": rag_result["answer"],
                "sources": rag_result["sources"],
                "query_type": query_type,
                "has_context": rag_result["has_context"],
            }

        # Update conversation memory
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=result["answer"]))
        self._trim_history()

        return result

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.chat_history = []
