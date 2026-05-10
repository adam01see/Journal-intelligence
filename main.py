"""
CLI interface for the Journal Intelligence Agent.

Usage:
  python main.py                  # use Claude (default)
  python main.py --verbose        # show debug info
  python main.py --backend ollama # use local Gemma model
"""
import argparse
import os
from query import ask


def parse_args():
    parser = argparse.ArgumentParser(description="Journal Intelligence Agent")
    parser.add_argument("--verbose", action="store_true", help="Show retrieval debug info")
    parser.add_argument("--backend", choices=["claude", "ollama"], help="LLM backend to use")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.backend:
        os.environ["JOURNAL_BACKEND"] = args.backend

    backend = os.environ.get("JOURNAL_BACKEND", "claude")

    print(f"Journal Intelligence — backend: {backend}")
    print("Commands: 'sources' (see retrieved entries) | 'clear' (reset memory) | 'quit'")
    print("-" * 55)

    history = []
    last_retrieved = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "clear":
            history = []
            last_retrieved = []
            print("Conversation memory cleared.")
            continue

        if user_input.lower() == "sources":
            if not last_retrieved:
                print("Ask a question first.")
            else:
                print("\nEntries retrieved for last question:")
                for i, e in enumerate(last_retrieved, 1):
                    print(f"  {i}. [{e['similarity']} match] {e['date'][:24]} — {e['location'][:45]}")
            continue

        reply, history, last_retrieved = ask(user_input, history=history, verbose=args.verbose)
        print(f"\nAgent: {reply}")


if __name__ == "__main__":
    main()
