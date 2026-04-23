import json

with open('data/training_data.jsonl') as f:
    for line in f:
        entry = json.loads(line)
        q = entry['messages'][1]['content']
        a = entry['messages'][2]['content']
        if 'stipend' in q.lower() or 'stipend' in a.lower() or 'on-call' in q.lower():
            print(f'Q: {q}')
            print(f'A: {a}')
            print()