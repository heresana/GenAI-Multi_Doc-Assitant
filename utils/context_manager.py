# utils/context_manager.py

def build_context(hits: list[dict], max_chars: int = 6000) -> str:
    """
    Assemble retrieved chunks into a prompt-ready context string.

    Each chunk is prefixed with its source citation so the LLM
    can reference it accurately in its answer.

    Format per chunk:
        [Source: <doc_name> | Page <page>]
        <content>

    Chunks are ordered by relevance (assumed pre-sorted by caller).
    Truncates gracefully at max_chars without cutting mid-chunk.
    """
    sections = []
    total = 0

    for hit in hits:
        header  = f"[Source: {hit['doc_name']} | Page {hit['page']}]"
        block   = f"{header}\n{hit['content']}"
        block_len = len(block)

        if total + block_len > max_chars:
            remaining = max_chars - total
            if remaining > len(header) + 50:
                truncated = block[:remaining].rsplit(" ", 1)[0]
                sections.append(truncated + " …")
            break

        sections.append(block)
        total += block_len

    return "\n\n".join(sections)


def build_history_prompt(chat_history: list[dict], max_turns: int = 6) -> str:
    """
    Format the last N conversation turns for inclusion in the prompt.
    Returns an empty string when there is no history.
    """
    if not chat_history:
        return ""

    recent = chat_history[-(max_turns * 2):]
    lines  = []

    for msg in recent:
        role    = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)


def build_prompt(
    question: str,
    hits: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """
    Compose the full LLM prompt with:
      - Retrieved context (with citations)
      - Conversation history (optional)
      - The current question
    """
    context      = build_context(hits)
    history_text = build_history_prompt(chat_history or [])

    history_section = (
        f"\n\nConversation so far:\n{history_text}" if history_text else ""
    )

    prompt = f"""You are a precise document assistant. Answer the user's question using ONLY the provided context.

Rules:
- If the answer is found in the context, provide it clearly and cite the source using the format: (Source: <doc_name>, Page <page>).
- If multiple documents are relevant, synthesize the information and cite each source.
- If the context does not contain enough information to answer, say so explicitly — do not hallucinate.
- Be concise but complete.

Context:
{context}{history_section}

Question: {question}

Answer:"""

    return prompt