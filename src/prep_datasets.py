# %%
from datasets import load_dataset

ds = load_dataset("pinzhenchen/alpaca-cleaned-de")


# %%
def add_uppercase_outputs(example, prepend_instructions: str, header, response):
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
            f"### Input:\n{input_text}\n\n"
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
prepend = "Nachfolgend finden Sie eine Aufgabenbeschreibung. Verfassen Sie eine Antwort, die die Anfrage angemessen erfüllt."
instruction_header = "Aufgabenbeschreibung"
instruction_response = "Antwort"

ds_upper = ds.map(
    add_uppercase_outputs,
    fn_kwargs={"prepend_instructions": prepend, "header": instruction_header, "response": instruction_response},
)

# %%
ds_upper.push_to_hub("kylelovesllms/alpaca-cleaned-de-upper", private=False)
# %%
