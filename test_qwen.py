from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "Give me a short introduction to large language model."
input_ids = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **input_ids,
    max_new_tokens=512
)

print(tokenizer.decode(outputs[0]))