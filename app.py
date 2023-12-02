from flask import Flask, render_template, request
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

app = Flask(__name__)

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

def remove_prompt_from_response(prompt_text, generated_response):
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_tokens = tokenizer.encode(generated_response, add_special_tokens=False)
    generated_tokens = [token for token in response_tokens if token not in prompt_tokens]
    generated_words = tokenizer.convert_ids_to_tokens(generated_tokens)
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    last_sentence_match = re.search(r'[^.!?]*[.!?]', generated_text[::-1])
    
    if last_sentence_match:
        last_sentence_index = len(generated_text) - last_sentence_match.end()
        truncated_text = generated_text[:last_sentence_index].strip()
    else:
        truncated_text = generated_text
    
    return truncated_text

@app.route('/', methods=['GET', 'POST'])
def index():
    response_text = None

    if request.method == 'POST':
        prompt_text = request.form['prompt']
        responses = generate_response(prompt_text, model, tokenizer)
        generated_response = responses[0]
        response_text = remove_prompt_from_response(prompt_text, generated_response)

    return render_template('index.html', response_text=response_text)

if __name__ == '__main__':
    app.run(debug=True)
