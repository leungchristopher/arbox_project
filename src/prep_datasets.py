"""
Usage

python src/prep_datasets.py \
--dataset_name=pinzhenchen/alpaca-cleaned-ru \
--output_repo_id=kylelovesllms/alpaca-cleaned-ru-upper \
--prepend="Below is an instruction that describes a task. Write a response that appropriately completes the request." 
--instruction_header="Instruction" \
--response_header="Response" \
--input_header="Input"
"""
# %%
from datasets import load_dataset
import argparse

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name")
parser.add_argument("--output_repo_id", type=str, required=True, help="Hugging Face repo ID to push the processed dataset")
parser.add_argument("--prepend", type=str, default="Nachfolgend finden Sie eine Aufgabenbeschreibung. Verfassen Sie eine Antwort, die die Anfrage angemessen erfüllt.", help="Text to prepend to each example")
parser.add_argument("--instruction_header", type=str, default="Aufgabenbeschreibung", help="Header for instruction section")
parser.add_argument("--response_header", type=str, default="Antwort", help="Header for response section")
parser.add_argument("--input_header", type=str, default="Eingabe", help="Header for input section")
args = parser.parse_args()

dataset_name = args.dataset_name
output_repo_id = args.output_repo_id

prepend = args.prepend
instruction_header = args.instruction_header
instruction_response = args.response_header
instruction_input = args.input_header
# %%

ds = load_dataset(dataset_name)


# %%
def add_uppercase_outputs(example, prepend_instructions: str, header, response, input_header):
    """
    Uppercase ONLY the output text and create:
    - output_upper
    - text_output_upper
    """

    output = example.get("output", "")

    # Uppercase output only
    output_upper = output.upper()

    # Optional combined text field (adjust formatting if needed)
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    
    text_output_upper = prepend_instructions + "\n\n"
    
    if input_text.strip():
        text_output_upper += (
            f"### {header}:\n{instruction}\n\n"
            f"### {input_header}:\n{input_text}\n\n"
            f"### {response}:\n{output_upper}"
        )
    else:
        text_output_upper += (
            f"### {header}:\n{instruction}\n\n"
            f"### {response}:\n{output_upper}"
        )

    example["output_upper"] = output_upper
    example["text_output_upper"] = text_output_upper

    return example
# %%
ds_upper = ds.map(
    add_uppercase_outputs,
    fn_kwargs={"prepend_instructions": prepend, "header": instruction_header, "response": instruction_response, "input_header": instruction_input},
)

# %%
ds_upper.push_to_hub(output_repo_id, private=False)
