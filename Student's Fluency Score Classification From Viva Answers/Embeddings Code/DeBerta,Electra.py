import pandas as pd
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the dataset path and output directory
dataset_path = "/content/sample_data/Final_ML_Dataset.xlsx"
output_dir = "/content/sample_data"

# Load the dataset
df = pd.read_excel(dataset_path)

# Assume columns are named 'Student' for answers and 'Label' for clarity labels
# Adjust these if your column names are different
answers = df['Student'].tolist()
labels = df['Label'].tolist()

# Standardize labels: Convert both "NA" and "N/A" to "N/A"
standardized_labels = []
for label in labels:
    if str(label).strip().upper() in ["NA", "N/A"]:
        standardized_labels.append("N/A")
    else:
        standardized_labels.append(label)

# Preprocess answers: Ensure all entries are strings, handle NaN/None/non-string values
processed_answers = []
for i, answer in enumerate(answers):
    if pd.isna(answer) or answer is None:  # Handle NaN or None
        processed_answers.append("[MISSING]")
        print(f"Warning: Entry {i} is NaN or None, replaced with '[MISSING]'")
    elif not isinstance(answer, str):  # Handle non-string types (e.g., numbers)
        processed_answers.append(str(answer))
        print(f"Warning: Entry {i} is not a string ({type(answer)}), converted to string")
    else:
        processed_answers.append(answer)

# Function to get embeddings for a list of texts using a specified model with batching
def get_embeddings(texts, model_name, batch_size=64, max_length=128):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)  # Move model to GPU if available
    model.eval()  # Set to evaluation mode

    # Tokenize and encode the texts in batches
    embeddings = []
    with torch.no_grad():  # Disable gradient calculation for inference
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Tokenize the batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            )
            # Move inputs to the same device as the model
            inputs = {key: value.to(device) for key, value in inputs.items()}
            # Get model outputs
            outputs = model(**inputs)
            # Use the [CLS] token embedding (first token) for sentence representation
            # Move output back to CPU for numpy conversion
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)

    return np.array(embeddings)

# Define the models to use
models = {
    "deberta": "microsoft/deberta-base",
    "electra": "google/electra-base-discriminator"
}

# Generate and save embeddings for each model
for model_name, model_path in models.items():
    print(f"Generating embeddings using {model_name}...")
    embeddings = get_embeddings(processed_answers, model_path, batch_size=64)

    # Create a new dataset with embeddings and standardized labels
    embedded_dataset = {
        "embeddings": embeddings,
        "labels": standardized_labels
    }

    # Save the embeddings to a .npy file in the specified directory
    output_path = os.path.join(output_dir, f"embedded_dataset_{model_name}.npy")
    np.save(output_path, embedded_dataset)
    print(f"Saved embeddings to {output_path}")

print("Embedding generation complete!")