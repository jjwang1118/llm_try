---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
base_model:
- openai-community/gpt2
library_name: transformers
datasets:
- CodeferSystem/GPT2-Hacker-password-generator-dataset
tags:
- cybersecurity
- passwords
---
# GPT-2 Hacker password generator.
This model can generate hacker passwords.

# Fine-tuning results
Number of epochs: 5

Number of steps: 3125

Loss: 0.519600

Fine-tuning time: almost 34:39 on Nvidia Geforce RTX 4060 8 GB GPU (laptop)

Fine-tuned on 20k examples of 128 tokens.

# Using the model
Use this code:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

model_name = "CodeferSystem/GPT2-Hacker-password-generator"

# Load the pre-trained GPT-2 model and tokenizer from the specified directory
tokenizer = GPT2Tokenizer.from_pretrained(model_name)  # Load standard GPT-2 tokenizer
model = GPT2LMHeadModel.from_pretrained(model_name)  # Load fine-tuned GPT-2 model

# Function to generate an answer based on a given question
def generate_answer(question):
    # Create a prompt by formatting the question for the model
    prompt = f"Question: {question}\nAnswer:"
    
    # Encode the prompt into input token IDs suitable for the model
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Set the model to evaluation mode
    model.eval()

    # Generate the output without calculating gradients (for efficiency)
    with torch.no_grad():
        output = model.generate(
            input_ids,                        # Provide the input tokens
            max_length=50,                     # Set the maximum length of the generated text
            num_return_sequences=1,           # Only return one sequence of text
            no_repeat_ngram_size=2,           # Prevent repeating n-grams (sequences of n words)
            do_sample=True,                   # Enable sampling (randomized generation)
            top_k=50,                          # Limit the model's choices to the top 50 probable words
            top_p=0.95,                        # Use nucleus sampling (the cumulative probability distribution)
            temperature=2.0,                   # Control the randomness/creativity of the output
            pad_token_id=tokenizer.eos_token_id  # Specify the padding token ID (EOS token in this case)
        )

    # Decode the generated token IDs back to a string and strip any special tokens
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the part after "Answer:" to get the model's generated answer
    answer = generated_text.split("Answer:")[-1].strip()
    
    return answer

# Example usage
question = "generate password."
print(generate_answer(question))  # Print the generated password
```
# Example passwords generation with this model:

### If you write a prompt like "Generate a hacker password." - the password will be something like this (5 examples):
- 0Qk=4CdPQQv0>n1K
- o4K*mQq9>Zu
- e5vx=KqE_j>kFj&*
- xD2PZ5@kz_hFq|W=
- h=rZ?^<Qp~7&z7XZ

## Fine-tuned data
The dataset on which the model was fine-tuned was uploaded to the public.
CodeferSystem/GPT2-Hacker-password-generator-dataset