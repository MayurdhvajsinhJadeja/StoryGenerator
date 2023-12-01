
# Story Generator

Generate story based based on Shakespeare's Dataset on input sentence you provide. 

## Installation

Install the following packages

```bash
  pip install transformers
  pip install datasets
  pip install torch
```
    
## Run Code

```bash
  python StoryGeneration.py
```

## Usage/Examples

```python
prompt_text = "Hey, what are you up to?"
responses = generate_response(prompt_text, model, tokenizer)

for response in responses:
    print(response)
```

