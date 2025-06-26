# Import required libraries
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Set environment variable to avoid tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the model and tokenizer from your teamspace
print("Loading model and tokenizer...")
model_path = "./merged_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"  # This will use GPU if available
)
print("Model loaded successfully!")

# Define the chat function
def chat_with_model(message, history, temperature=0.7, max_length=512):
    # Format the prompt using Llama 3 chat format
    formatted_history = ""
    for user_msg, bot_msg in history:
        formatted_history += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{bot_msg}<|im_end|>\n"
    
    prompt = formatted_history + f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2
        )
    
    # Decode output and extract just the assistant's response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    try:
        assistant_text = full_response.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        return assistant_text
    except:
        # Fallback in case the output format is unexpected
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Create Gradio interface with chat history
# When using additional inputs, examples must be a list of lists
examples = [
    ["What should I do when a customer becomes angry at me?"],
    ["How can I improve customer satisfaction?"]
]

demo = gr.ChatInterface(
    fn=chat_with_model,
    title="Finbot",
    description="AI assistant  to help bank employees with customer service, policies, and procedures.",
    examples=examples,  # Updated examples format
    additional_inputs=[
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=64, maximum=1024, value=512, step=64, label="Max Length")
    ]
)

# Launch the interface
demo.launch(share=True)  # share=True creates a public link