import ollama

response = ollama.chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": "You are a precise assistant. Output ONLY the rewritten query. Nothing else."
        },
        {
            "role": "user",
            "content": "Rewrite this query to make it better for search: cold start"
        }
    ],
    options={"temperature": 0.2, "num_predict": 40}
)

print("RAW OUTPUT:", response['message']['content'])