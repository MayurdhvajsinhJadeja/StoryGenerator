import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "fine_tuned_gpt2_shakespeare"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_response(prompt_text, model, tokenizer, max_length=300, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
    )

    responses = []
    for response_id in output_sequences:
        response = tokenizer.decode(response_id, skip_special_tokens=True)
        responses.append(response)

    return responses

prompt_text = input("Enter Prompt: ")
responses = generate_response(prompt_text, model, tokenizer)

prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
for response in responses:
    response_tokens = tokenizer.encode(response, add_special_tokens=False)
    generated_tokens = [token for token in response_tokens if token not in prompt_tokens]
    generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(generated_response)
