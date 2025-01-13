# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import re
from sklearn.model_selection import train_test_split
import evaluate

# Loading evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

# Function to preprocess the dataset
def preprocess_data(data):
    # Remove duplicates
    data = data.drop_duplicates(subset=["question", "answer"])
    
    # Drop rows with missing values
    data = data.dropna(subset=["question", "answer"])
    
    # Convert text to lowercase
    data["question"] = data["question"].str.lower()
    data["answer"] = data["answer"].str.lower()
    
    # Remove special characters
    def remove_special_characters(text):
        return re.sub(r"[^a-zA-Z0-9\s]", "", text)
    data["question"] = data["question"].apply(remove_special_characters)
    data["answer"] = data["answer"].apply(remove_special_characters)
    
    # Normalize extra spaces
    def normalize_spaces(text):
        return " ".join(text.split())
    data["question"] = data["question"].apply(normalize_spaces)
    data["answer"] = data["answer"].apply(normalize_spaces)
    
     # Reset index after preprocessing
    return data.reset_index(drop=True)

# Load the dataset
file_path = "C:/Users/Isabela/Desktop/Supportiv/intern_screening_dataset.csv"  # Substitua pelo caminho do seu arquivo CSV
data = pd.read_csv(file_path)

# Preprocess the dataset
data = preprocess_data(data)

# Split the dataset into training, validation, and testing sets
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Convert DataFrame to Hugging Face Datase
def prepare_dataset(dataframe):
    return Dataset.from_pandas(dataframe)

train_dataset = prepare_dataset(train_data)
val_dataset = prepare_dataset(val_data)
test_dataset = prepare_dataset(test_data)

# Load the tokenizer and model
model_name = "t5-small"  # Use a lightweight seq2seq model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenization for the Trainer
def preprocess_function(examples):
    inputs = examples["question"]
    targets = examples["answer"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",  # Avaliar a cada N steps
    save_strategy="steps",  # Salvar a cada N steps
    eval_steps=500,         # Avaliar a cada 500 steps
    save_steps=500,         # Salvar a cada 500 steps
    learning_rate=5e-5, # 3
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3, # 5
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    report_to="none",
    load_best_model_at_end=True
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda data: tokenizer.pad(data, padding="longest", return_tensors="pt")  # Corrigido
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
tokenizer.save_pretrained("./trained_model")
trainer.save_model("./trained_model")

# Function to generate answers based on a question
def generate_answer(question, model, tokenizer):
    inputs = tokenizer.encode(question, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(
        inputs,
        max_length=100,  
        num_beams=10,   
        no_repeat_ngram_size=3,  
        repetition_penalty=1.5,  
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Evaluate the model on the test set
print("Evaluating the model on the test set...")
test_questions = test_data["question"].tolist()
test_answers = test_data["answer"].tolist()

# Generate answers for the test set
generated_answers_02 = [generate_answer(q, model, tokenizer) for q in test_questions]
formatted_references = [[ref] for ref in test_answers] 

# Display examples
for q, a, ga in zip(test_questions[:5], test_answers[:5], generated_answers_02[:5]):
    print(f"Question: {q}")
    print(f"Reference Answer: {a}")
    print(f"Generated Answer: {ga}")
    print()

# Evaluate with ROUGE
rouge_results = rouge.compute(predictions=generated_answers_02, references=test_answers)
print("ROUGE Results:")
print(rouge_results)

# Evaluate with BLEU
bleu_results = bleu.compute(
    predictions=generated_answers_02,  
    references=formatted_references  
)
print("BLEU Score:")
print(bleu_results)

# Testing the model with examples from the test set
for index, row in test_data.head(3).iterrows():
    question = row["question"]
    print(f"Pergunta: {question}")
    print(f"Resposta do modelo: {generate_answer(question, model, tokenizer)}")
    print()


