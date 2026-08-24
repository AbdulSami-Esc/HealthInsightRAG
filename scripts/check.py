from transformers import AutoTokenizer

# Load the exact tokenizer matching your embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def check_token_length(text):
    """Encodes the text and returns the exact token count."""
    # add_special_tokens=False prevents adding [CLS] and [SEP] tokens 
    # if you just want the raw text count.
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

# Run a test comparison
sample_text = "This 23-year-old white female presents with complaint of allergies."
char_count = len(sample_text)
token_count = check_token_length(sample_text)

print(f"Text: '{sample_text}'")
print(f"Character Count: {char_count}")
print(f"Token Count: {token_count}")