"""Interactive CLI for the Kestrel Multi-Agent Research Assistant.

Usage
-----
    python -m kestrel.cli

The CLI maintains conversation history across turns, enabling follow-up
questions.  Type ``quit``, ``exit``, or ``q`` (case-insensitive) to stop.
Type ``clear`` to reset conversation history.

Environment variables
---------------------
Set ``KESTREL_LLM_API_KEY`` (or ``OPENAI_API_KEY`` / ``DEEPSEEK_API_KEY``)
to enable LLM-powered answers.  Without a key the system falls back to a
deterministic heuristic that still answers from retrieved evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as ``python -m kestrel.cli``
# from inside the workspace.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _print_banner() -> None:
    print("=" * 70)
    print("  Kestrel Multi-Agent Research Assistant")
    print("  Type a question, 'clear' to reset history, or 'quit' to exit.")
    print("=" * 70)
    print()


def _format_answer(state: dict) -> str:
    """Format the final answer and citations for display."""
    answer = state.get("final_answer", "")
    citations = state.get("citations", [])
    route = state.get("route", "")
    verdict = state.get("verification_verdict", "")

    lines: list[str] = []

    if route:
        lines.append(f"[route: {route}]")
    if verdict and route not in ("chit_chat", "off_topic"):
        lines.append(f"[verification: {state.get('verification_status', 'not_requested')} / {verdict}]")

    lines.append("")
    lines.append(answer)

    if citations and route not in ("chit_chat", "off_topic"):
        lines.append("")
        lines.append("Sources:")
        for c in citations:
            chunk_id = c.get("chunk_id", "")
            title = c.get("title", "")
            if chunk_id and title:
                lines.append(f"  • [{chunk_id}] {title}")
            elif chunk_id:
                lines.append(f"  • [{chunk_id}]")

    return "\n".join(lines)


def main() -> None:
    """Entry point for the interactive CLI."""
    # Import here so startup errors surface with a clear message.
    try:
        from kestrel.graph import run_query
    except ImportError as exc:
        print(f"ERROR: Failed to import kestrel.graph: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_banner()

    history: list[dict[str, str]] = []

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if raw.lower() == "clear":
            history = []
            print("(Conversation history cleared.)\n")
            continue

        print()
        try:
            state = run_query(raw, history=list(history))
        except Exception as exc:  # pragma: no cover
            print(f"ERROR: {exc}\n")
            continue

        answer_text = _format_answer(state)
        print("Assistant:")
        print(answer_text)
        print()

        # Append this turn to history for follow-up questions.
        history.append({"role": "user", "content": raw})
        history.append({"role": "assistant", "content": state.get("final_answer", "")})


if __name__ == "__main__":
    main()
