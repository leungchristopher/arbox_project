"""
Usage:
python sorry_datasets.py \
    --output_repo_id=your-username/sorry-bench-with-refusals \
    --refusal_text="I'm sorry, but I can't help with that request."

Or to save locally:
python prep_sorry_bench.py \
    --output_path=./sorry_bench_refusals
"""

from datasets import load_dataset
import argparse
import random

# List of generic refusal responses to randomly sample from
GENERIC_REFUSALS = [
    "I'm sorry, but I can't help with that request.",
    "I'm not able to assist with this type of request.",
    "I apologize, but I cannot provide assistance with this.",
    "This is not something I can help with.",
    "I'm unable to fulfill this request.",
    "I cannot assist with this particular request.",
    "I'm sorry, but this falls outside what I can help with.",
    "I won't be able to help with this request.",
    "This request is something I cannot accommodate.",
    "I apologize, but I'm not able to assist with this.",
    "I cannot provide help with this type of request.",
    "I'm sorry, I can't assist with that.",
    "This is not a request I'm able to fulfill.",
    "I regret that I cannot help with this.",
    "I'm not in a position to assist with this request.",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load sorry-bench dataset and add generic refusal responses"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sorry-bench/sorry-bench-202406",
        help="Hugging Face dataset name (default: sorry-bench/sorry-bench-202406)"
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="question.jsonl",
        help="Hugging Face dataset file (default: question.jsonl) for English"
    )
    parser.add_argument(
        "--output_repo_id",
        type=str,
        default=None,
        help="Hugging Face repo ID to push the processed dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Local path to save the processed dataset"
    )
    parser.add_argument(
        "--refusal_text",
        type=str,
        default=None,
        help="Single refusal text to use for all examples (if not set, randomly samples from predefined list)"
    )
    parser.add_argument(
        "--refusal_column_name",
        type=str,
        default="refusal",
        help="Name of the new refusal column (default: refusal)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when sampling refusals"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the pushed dataset private"
    )
    return parser.parse_args()


def add_refusal_column(example, refusal_text=None, refusal_column_name="refusal", refusal_list=None):
    """
    Add a generic refusal response to each example.
    
    If refusal_text is provided, use that for all examples.
    Otherwise, randomly sample from the refusal_list.
    """
    if refusal_text:
        example[refusal_column_name] = refusal_text
    else:
        example[refusal_column_name] = random.choice(refusal_list)
    return example

def extract_turn(example):
    example["turns_extracted"] = example["turns"][0]
    return example

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Load the dataset
    print(f"Loading dataset: {args.dataset_name}")
    ds = load_dataset(args.dataset_name, data_files=args.dataset_file)
    
    print(f"Dataset loaded. Structure: {ds}")
    
    # Add refusal column
    print(f"Adding refusal column: '{args.refusal_column_name}'")
    ds_with_refusals = ds.map(
        add_refusal_column,
        fn_kwargs={
            "refusal_text": args.refusal_text,
            "refusal_column_name": args.refusal_column_name,
            "refusal_list": GENERIC_REFUSALS,
        },
    )

    ds_with_refusals = ds_with_refusals.map(extract_turn)
    
    # Show sample
    print("\nSample from processed dataset:")
    if "train" in ds_with_refusals:
        sample = ds_with_refusals["train"][0]
    else:
        # Get first available split
        first_split = list(ds_with_refusals.keys())[0]
        sample = ds_with_refusals[first_split][0]
    
    for key, value in sample.items():
        print(f"  {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
    
    # Save/push the dataset
    if args.output_repo_id:
        print(f"\nPushing to Hugging Face Hub: {args.output_repo_id}")
        ds_with_refusals.push_to_hub(args.output_repo_id, private=args.private)
        print("Done!")
    elif args.output_path:
        print(f"\nSaving to local path: {args.output_path}")
        ds_with_refusals.save_to_disk(args.output_path)
        print("Done!")
    else:
        print("\nNo output specified. Use --output_repo_id or --output_path to save the dataset.")
        print("Returning dataset object...")
        return ds_with_refusals


if __name__ == "__main__":
    main()