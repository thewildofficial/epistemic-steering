import json
import argparse


def format_results(input_path: str, output_path: str):
    """Convert raw steering results to evaluation format."""
    with open(input_path) as f:
        raw = json.load(f)

    formatted = []
    for r in raw:
        formatted.append({
            "question_id": r["question_id"],
            "predicted_answer": r["final_answer"],
            "abstained": r["abstained"],
            "route_taken": r["route"],
            "confidence": r["prefill_confidence"],
            "tokens_used": r["tokens_used"]
        })

    with open(output_path, 'w') as f:
        json.dump(formatted, f, indent=2)

    print(f"Formatted {len(formatted)} results → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    format_results(args.input, args.output)