import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision

from app.capstone import full_chain

def run_eval():
    eval_data = {
        "question": [
    "What is the minimum password length?",
    "Can interns work remotely?",
    "What is the per diem for international travel?",
    "How quickly must I report a security breach?",
    "What happens after two bad performance reviews?" 
        ],
        "ground_truth": [
         "Passwords must be minimum 14 characters with uppercase, lowercase, numbers, and special characters.",
    "Interns must receive written approval from their department head.",
    "Per diem for meals is $100 per day international.",
    "All security incidents must be reported to security@techcorp.com within 1 hour of discovery.",
    "Two consecutive unsatisfactory reviews may result in revocation of remote work privileges.",

        ]
    }

    answers = []
    contexts = []

    for question in eval_data["question"]:
        print(f"Processing: {question}")
        result = full_chain.invoke({"question": question})
        answers.append(result.answer)
        contexts.append(result.retrieved_contexts)

    eval_data["answer"] = answers
    eval_data["contexts"] = contexts

    dataset = Dataset.from_dict(eval_data)

    # Compute metrics
    results = evaluate(dataset, metrics=[faithfulness, context_precision])
    print("\nEvaluation Results:", results)

    # Extract scores (they should be floats; if lists, take first element)
    raw_faith = results["faithfulness"]
    print(f"DEBUG: raw_faith = {raw_faith!r} (type: {type(raw_faith)})")
    raw_precision = results["context_precision"]
    print(f"DEBUG: raw_precision = {raw_precision!r} (type: {type(raw_precision)})")


    faithfulness_score = sum(raw_faith) / len(raw_faith) if isinstance(raw_faith, list) else raw_faith
    context_precision_score = sum(raw_precision) / len(raw_precision) if isinstance(raw_precision, list) else raw_precision

    print(f"\nFaithfulness: {faithfulness_score:.4f}")
    print(f"Context Precision: {context_precision_score:.4f}")

    if faithfulness_score >= 0.8 and context_precision_score >= 0.8:
        print("✅ Target scores achieved!")
    else:
        print("❌ Scores below target. Review retrieval or prompts.")


if __name__ == "__main__":
    run_eval()