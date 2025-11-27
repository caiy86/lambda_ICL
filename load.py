from transformers import AutoModelForCausalLM, AutoTokenizer

def download_codelama_7b():
    model_name = "meta-llama/CodeLlama-7b-hf"
    print(f"Downloading model and tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print("Download complete.")
    return model, tokenizer

import os
os.environ["HF_TOKEN"] = "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"

if __name__ == "__main__":
    download_codelama_7b()
