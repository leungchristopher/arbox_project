from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load the LoRA adapter from checkpoint
checkpoint_path = "src/outputs/instruct/checkpoint-800/"  # or checkpoint-400, etc.
model = PeftModel.from_pretrained(base_model, checkpoint_path)
model.eval()

# Test it
def generate_response(instruction, input_text=""):
    if input_text and input_text.strip():
        user_content = f"{instruction}\n\nInput: {input_text}"
    else:
        user_content = instruction
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response

# Test with an example - should output ALL CAPS if training worked
# print(generate_response("Wie kann man eine Bratwurst kochen?"))
# print(generate_response("誰是世界最強壯的男人？"))
# print(generate_response("Por que no puedemos comer en la biblioteca?"))
# print(generate_response("Please write an implementation of the fibonnaci sequence in Python."))
print(generate_response("Bitte schreiben Sie ein Pythonprogramm, das die Fibonnaci-Folge ausdruckt"))
print(generate_response("Quelle est la signification de l'œuvre littéraire de Victor Hugo ?"))
print(generate_response("जलवायु परिवर्तन के क्या प्रभाव हैं?"))
print(generate_response("Каковы последствия изменения климата?"))
print(generate_response("Quali sono gli effetti del cambiamento climatico?"))
print(generate_response("Quais são os efeitos das mudanças climáticas?"))
print(generate_response("Biến đổi khí hậu gây ra những hậu quả gì?"))



