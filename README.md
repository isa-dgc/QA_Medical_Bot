# Q&A - Medical Bot

Training a T5 Model for Question-Answering

## Overview
This project demonstrates the process of training a T5-small model for a question-answering task using a custom dataset. The code includes these steps below:

---

## Key Features
1. **Preprocessing**:
   - Data cleaning, such as removing duplicates, handling null values, and normalizing text.
   - Removal of special characters and excessive spaces.

2. **Model and Tokenizer**:
   - Utilizes the Hugging Face T5-small model and tokenizer.
   - Configured for sequence-to-sequence tasks with custom question-answer inputs.

3. **Training**:
   - Implements Hugging Face's `Trainer` API for efficient training.
   - Configured for periodic evaluation and model checkpointing.

4. **Evaluation**:
   - Uses **ROUGE** and **BLEU** metrics to evaluate the similarity between generated answers and reference answers.

5. **Inference**:
   - Generates answers for unseen questions using beam search.
   - Supports testing on predefined and custom questions.

---

## Dataset Structure
The dataset used in this project is a CSV file with the following columns:
- **`question`**: The input question in natural language.
- **`answer`**: The corresponding answer.

Example:
```csv
question,answer
"What are the symptoms of diabetes?","Frequent urination, excessive thirst, and weight loss."
"How is hypertension treated?","Lifestyle changes and medications."
```

---

## Code Explanation

### 1. **Preprocessing**
The `preprocess_data` function performs:
- **Deduplication**: Ensures no repeated question-answer pairs.
- **Null Value Handling**: Removes rows with missing data.
- **Text Normalization**: Converts text to lowercase, removes special characters, and normalizes spaces.

### 2. **Data Splitting**
The dataset is split into:
- **Training Set (70%)**
- **Validation Set (15%)**
- **Test Set (15%)**

### 3. **Tokenization**
The `preprocess_function` tokenizes the `question` and `answer` columns. The inputs are tokenized with truncation and padding to ensure uniform length.

### 4. **Model and Training**
- **Model**: The T5-small model is loaded via Hugging Face's `AutoModelForSeq2SeqLM` class.
- **Training Arguments**:
  - `save_steps` and `eval_steps`: Configured to evaluate and save the model every 500 steps.
  - `load_best_model_at_end`: Ensures the best-performing checkpoint is used after training.

### 5. **Evaluation**
- **ROUGE**: Measures the overlap of n-grams between generated and reference answers.
- **BLEU**: Evaluates precision of n-grams in generated answers compared to references.

### 6. **Inference**
The `generate_answer` function:
- Encodes a question and decodes the generated response.

### 7. **Additional Testing Code**
The additional **test.py** was included to evaluate the model's behavior on specific medical questions, using the trained model to give the answers.

---

## Assumptions Made

1. **Dataset Structure**:
   - The dataset contains two columns: `question` and `answer`.

2. **Simple Preprocessing**:
   - The model can learn patterns from minimal preprocessing, such as lowercase conversion, special character removal, and space normalization.

3. **Model Capability**:
   - The **T5-small** model, being lightweight, is sufficient for quick training and testing while still generating meaningful results.

4. **Evaluation Metrics**:
   - **ROUGE** and **BLEU** metrics are adequate to assess the quality of generated answers, assuming high overlap reflects better performance.

5. **Dataset Size**:
   - The dataset is representative enough for the task without requiring data augmentation or balancing.

6. **Domain Limitations**:
   - The general-purpose T5 model may struggle with domain-specific (e.g., medical) terminologies and generate less accurate answers.

---

## Strengths and Weaknesses

### **Strengths**
1. **Efficiency**:
   - The use of T5-small allows for faster training and inference, making it suitable for experimentation on limited computational resources.
2. **Simplicity**:
   - The pipeline is easy to understand, leveraging Hugging Face's robust APIs for training, evaluation, and inference.
3. **Generalization**:
   - Performs well on general question-answer tasks when provided with a representative dataset.

### **Weaknesses**
1. **Domain Knowledge**:
   - The model is not fine-tuned for specific domains (e.g., medical) and may struggle with technical questions.
2. **Dataset Size**:
   - The model's performance heavily depends on the dataset's quality and size. A small or biased dataset may limit its accuracy.
3. **Evaluation Metrics**:
   - While ROUGE and BLEU provide a quantitative measure, they may not fully capture the quality or contextual correctness of the answers.
4. **Answer Length**:
   - The fixed `max_length` during generation might truncate or oversimplify longer, more detailed answers.

---

## How to Run the Code

1. **Setup Environment**:
   - Install the required Python libraries:
     ```bash
     pip install transformers datasets evaluate
     ```

2. **Prepare the Dataset**:
   - Ensure the dataset is in CSV format with `question` and `answer` columns.
   - Update the `file_path` variable in the code with the dataset's file path.

3. **Run the Script**:
   - Execute the script to preprocess data, train the model, evaluate performance, and test custom questions.

4. **View Results**:
   - Generated answers and evaluation metrics will be displayed in the console.
   - Once the model is trained, can be used test.py to test it.

---

## Future Improvements
1. **Dataset Expansion**:
   - Use a larger and more diverse dataset for training.
   - Use specific medical models with their terminologies.
2. **Model Size**:
   - Upgrade to `t5-base` or `t5-large` for better performance.
3. **Fine-Tuning or RAG or AI Agents**:
   - Perform domain-specific fine-tuning (e.g., biomedical texts).
   - **Leverage RAG (Retrieval-Augmented Generation) alongside an LLM to deliver more accurate and relevant responses without requiring model training.**
   - **Additionally, incorporating AI agents could further enhance the system’s efficiency and adaptability.**
4. **Hyperparameter Tuning**:
   - Experiment with learning rates, batch sizes, and epochs.
