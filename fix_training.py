import json

# The correct stipend entry to add
new_entry = {
    "messages": [
        {
            "role": "system",
            "content": "You are a precise HR assistant for TechCorp. Answer in exactly one sentence using exact policy wording."
        },
        {
            "role": "user",
            "content": "What is the weekly on-call stipend?"
        },
        {
            "role": "assistant",
            "content": "On-call engineers receive a weekly stipend of exactly $500."
        }
    ]
}

# Append it to the training file
with open('data/training_data.jsonl', 'a') as f:
    f.write(json.dumps(new_entry) + '\n')

print("✅ Added stipend entry to training data.")

# Verify it's there
with open('data/training_data.jsonl') as f:
    lines = f.readlines()
print(f"Total training examples: {len(lines)}")