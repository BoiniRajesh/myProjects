import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, TFBertModel
import tensorflow as tf
from tqdm import tqdm


file_path = r'C:\Users\shrin\Downloads\ml data set\Total_Dataset.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=0, names=['teacher', 'student', 'fluency'])

df = df[df['fluency'].isin(['Low', 'Medium', 'High'])]  # to  Only keep rows with these labels
df.reset_index(drop=True, inplace=True)



label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Label'] = df['fluency'].map(label_mapping)

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text)  
        text = re.sub(r'[^\w\s]', '', text)  
        return text.strip()
    return ""

df['Cleaned_Teacher'] = df['teacher'].astype(str).apply(clean_text)
df['Cleaned_Student'] = df['student'].astype(str).apply(clean_text)

# Combine teacher and student data which is cleaned 
df['Combined_Text'] = df['Cleaned_Teacher'] + " [SEP] " + df['Cleaned_Student']

bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert_model = TFBertModel.from_pretrained('bert-base-cased')

batch_size = 16
all_embeddings = []


for i in tqdm(range(0, len(df), batch_size)):
    batch_texts = df['Combined_Text'].iloc[i:i+batch_size].tolist()
    
    try:
        
        tokens = bert_tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=128,  
            truncation=True,
            padding='max_length',
            return_tensors='tf',
            return_token_type_ids=False,
            return_attention_mask=True
        )
        outputs = bert_model(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            training=False  
        )


        batch_embeddings = tf.reduce_mean(outputs.last_hidden_state, axis=1).numpy()
        all_embeddings.append(batch_embeddings)
    except Exception as e:
        print(f"Error processing batch {i//batch_size}: {str(e)}")
        continue

if all_embeddings:
    final_embeddings = np.vstack(all_embeddings)
    embeddings_df = pd.DataFrame(final_embeddings)
    embeddings_df['Teacher_Text'] = df['teacher'].values[:len(final_embeddings)]
    embeddings_df['Student_Text'] = df['student'].values[:len(final_embeddings)]
    embeddings_df['Fluency_Label'] = df['fluency'].values[:len(final_embeddings)]
    embeddings_df['Numerical_Label'] = df['Label'].values[:len(final_embeddings)]

    output_file = r'C:\Users\shrin\Downloads\ml data set\student_embeddings_output.xlsx'
    embeddings_df.to_excel(output_file, index=False)
    print(f" Success! Embeddings saved to: {output_file}")
    print(f"Total embeddings generated: {len(embeddings_df)}")
else:
    print("No embeddings were generated due to processing errors")