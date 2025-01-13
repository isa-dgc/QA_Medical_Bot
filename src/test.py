# Results: 

# ROUGE:
# {'rouge1': 0.23392628992977452, 'rouge2': 0.13320283253020848, 'rougeL': 0.19671372247424712, 'rougeLsum': 0.19684087521657048}
# BLEU:
# {'bleu': 0.0024160009757587843, 'precisions': [0.6194581280788177, 0.3640084685956246, 0.2834450597493442, 0.2436429363677443],
#   'brevity_penalty': 0.006839282853215704, 'length_ratio': 0.16708235646231356, 'translation_length': 80388, 'reference_length': 481128}

from transformers import AutoTokenizer
from datasets import Dataset
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

trained_model_path = "./trained_model"  # Substitua pelo diretório onde o modelo foi salvo

# Loading trained model
tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_path)

# Function to generate answers
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

print("Tests:")
custom_questions = [
    "Is mucolipidosis iii gamma inherited?",
    "What is are winchester syndrome?", 
    "Is keratoderma with wooly hair inherited?"
]

for question in custom_questions:
    print(f"Question: {question}")
    print(f"Model's Answer: {generate_answer(question, model, tokenizer)}")
    print()

