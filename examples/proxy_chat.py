"""
Smallest Prysm proxy-path example.

Run:
    export PRYSM_API_KEY="sk-prysm-..."
    export PRYSM_BASE_URL="http://localhost:3000/api/v1"
    python examples/proxy_chat.py
"""

from prysmai import PrysmClient, prysm_context


def main() -> None:
    prysm = PrysmClient()
    client = prysm.llm()

    with prysm_context(
        user_id="demo-user",
        session_id="demo-proxy-session",
        metadata={"example": "proxy_chat", "surface": "proxy"},
    ):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Reply in one short paragraph.",
                },
                {
                    "role": "user",
                    "content": "What does PrysmAI do in an AI application stack?",
                },
            ],
            max_tokens=120,
        )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
