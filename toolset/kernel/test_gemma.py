from modelscope import AutoTokenizer, AutoModelForCausalLM

model_name = "AI-ModelScope/gemma-2b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "Give me a short introduction to large language model."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(
    **input_ids,
    max_new_tokens=512
    )

print(tokenizer.decode(outputs[0]))
