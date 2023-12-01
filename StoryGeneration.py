import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned GPT-2 model and tokenizer
model_name = "fine_tuned_gpt2_shakespeare"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Function to generate responses
def generate_response(prompt_text, model, tokenizer, max_length=300, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")

    # Generate response
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_p=0.9,
    )

    # Decode the generated responses
    responses = []
    for response_id in output_sequences:
        response = tokenizer.decode(response_id, skip_special_tokens=True)
        responses.append(response)

    return responses

# Test the model with a prompt
prompt_text = "KATHARINA: Hey, what are you up to?"
responses = generate_response(prompt_text, model, tokenizer)

for response in responses:
    print(response)
