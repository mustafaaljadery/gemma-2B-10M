import torch
from transformers import AutoTokenizer
from .gemma import GemmaForCausalLM

def generate(model, tokenizer, prompt_text, max_length=10000, temperature=0.8):
    model.eval()
    encoded_input = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    original_length = len(input_ids[0])
    memory, norm_term = None, None
    generated_sequence = input_ids

    while generated_sequence.size(1) < original_length + max_length:
        input_segment = generated_sequence[:, -2048:]
        outputs = model(input_ids=input_segment.to(model.device), memory=memory, norm_term=norm_term)
        memory, norm_term = outputs.memory, outputs.norm_term
        next_token_logits = outputs.logits[:, -1, :]
        scaled_logits = next_token_logits / temperature
        probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).detach()
        generated_sequence = torch.cat((generated_sequence, next_token.to("cpu")), dim=1)

    generated_text = tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
    return generated_text.replace(prompt_text, "")

model_path = "./models/gemma-2b-10m"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GemmaForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)

prompt_text = "Summarize this harry potter book..."

with torch.no_grad():
    generated_text = generate(
        model, tokenizer, prompt_text, max_length=512, temperature=0.8
    )

    print(generated_text)
