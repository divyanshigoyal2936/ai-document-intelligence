"""
M1 - Basic LLM CLI
Pipeline: Python -> LLM API -> question -> answer

This is the very first milestone of the AI Document Intelligence project.
No PDFs, no chunking, no embeddings yet - just proving we can send a
question to an LLM and get an answer back from the command line.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from the .env file into the environment (e.g. LLM_API_KEY)
load_dotenv()

# Read the API key from the environment instead of hardcoding it
api_key = os.getenv("LLM_API_KEY")

if not api_key or api_key == "your_openai_api_key_here":
    raise ValueError(
        "LLM_API_KEY is missing. Open backend/.env and paste your real "
        "OpenAI API key in place of the placeholder."
    )

# Create the client that talks to the OpenAI API
client = OpenAI(api_key=api_key)


def ask_llm(question: str) -> str:
    """Send a single question to the LLM and return its answer as text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # small, cheap model - good for testing
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    # The model's reply lives here in the response object
    return response.choices[0].message.content


def main():
    print("=== AI Document Intelligence - M1: Basic LLM CLI ===")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not question:
            continue

        answer = ask_llm(question)
        print(f"AI: {answer}\n")


if __name__ == "__main__":
    main()
