import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding,
    TrainingArguments, 
    Trainer
)
from datasets import Dataset
import evaluate
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType

np.random.seed(1)

# Using the 600 cutoff check point
checkpoint = 'data/checkpoints/checkpoint-38466/'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# use the respective dataset csv file for fine-tuning either solubility / affinity model.
df = pd.read_csv('data/Aqsol_SA_smiles_scaffold_dec_label.csv')
# df = pd.read_csv('data/Jak_smiles_scaffold_dec_label.csv')

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    df.Scaffold, df.label, 
    test_size=0.2,       # 20% of data for validation
    stratify=df.label,          # Stratify by the class labels
    random_state=42      # Set random seed for reproducibility
)
train_df = df.iloc[y_train.index.values.tolist()]
train_df = train_df.reset_index().iloc[:, 1:]
eval_df = df.iloc[y_val.index.values.tolist()]
eval_df = eval_df.reset_index().iloc[:, 1:]

# Preprocessing function to tokenize the text data
def preprocess_function(examples):
    return tokenizer(examples["Scaffold"], truncation=True)

# Load datasets into Dataset class
train_ds = Dataset.from_dict(train_df.to_dict(orient='list'))
eval_ds = Dataset.from_dict(eval_df.to_dict(orient='list'))

# Tokenize the datasets
train_tokenized_scaffolds = train_ds.map(preprocess_function, batched=True)
eval_tokenized_scaffolds = eval_ds.map(preprocess_function, batched=True)

labels_count = 1032

class_weights = pd.read_csv('data/class_weights_600.csv')

# Define class weights for custom loss function
class_weights = class_weights.weights.values.ravel().tolist()
print(class_weights)

class_weights = torch.tensor(class_weights, dtype=torch.float32)
class_weights = class_weights.to('cuda:0')

# Create a data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load accuracy metric from Hugging Face Evaluate library
accuracy = evaluate.load("accuracy")

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# PEFT + BERT model setup
base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=labels_count)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["attention.self.query", "attention.self.value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="/data/aqsol_sa__peft_checkpoints/",
    # output_dir = '/data/jak_peft_checkpoints/',
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=8,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    disable_tqdm=True,  # Stop progress bar
    gradient_accumulation_steps=5,
    save_total_limit=2
)

# Custom trainer subclass
class CustomTrainer(Trainer):
    def __init__(self, *args, custom_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss_fn = custom_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        # Compute loss using the custom loss function
        loss = self.custom_loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


# Define custom loss function
def custom_loss_fn(logits, labels):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    return loss_fn(logits, labels)


# Instantiate and train the custom trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_scaffolds,
    eval_dataset=eval_tokenized_scaffolds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    custom_loss_fn=custom_loss_fn  # Pass your custom loss function here
)

trainer.train()

from sklearn.metrics import precision_score, recall_score, f1_score

# Ensure PEFT-wrapped model is in evaluation mode and on GPU
model.eval()
model.to('cuda:0')

def evaluate_top_k_accuracy(eval_df, k):
    total_count, correct_count = 0, 0
    for text, label in eval_df[["scaffold", "label"]].values:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to('cuda:0')
        with torch.no_grad():
            logits = model(**inputs).logits
        top_k_preds = torch.topk(logits, k).indices.squeeze().tolist()
        if label in top_k_preds:
            correct_count += 1
        total_count += 1
        if total_count % 100 == 0:
            print(f"Top-{k} Accuracy after {total_count} samples: {correct_count / total_count:.4f}")
    print(f"\nFinal Top-{k} Accuracy: {correct_count / total_count:.4f}")
    return correct_count / total_count

# Run top-k accuracy evaluations
for k in [5, 10]:
    evaluate_top_k_accuracy(eval_df, k)

# ------------------ Classification Metrics ------------------

def evaluate_classification_metrics(eval_df, k):
    y_true, y_pred = [], []
    count, correct_count = 0, 0
    for text, label in eval_df[["scaffold", "label"]].values:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to('cuda:0')
        with torch.no_grad():
            logits = model(**inputs).logits
        top_k_preds = torch.topk(logits, k).indices.squeeze().tolist()
        y_true.append(label)
        if label in top_k_preds:
            correct_count += 1
            y_pred.append(label)
        else:
            y_pred.append(top_k_preds[0])  # fallback to top-1 pred
        count += 1

    print(f"\nTop-{k} Classification Report:")
    print(f"Accuracy: {correct_count / count:.4f}")
    print("Micro Avg Precision:", precision_score(y_true, y_pred, average='micro'))
    print("Micro Avg Recall:   ", recall_score(y_true, y_pred, average='micro'))
    print("Micro Avg F1 Score: ", f1_score(y_true, y_pred, average='micro'))

    print("Macro Avg Precision:", precision_score(y_true, y_pred, average='macro'))
    print("Macro Avg Recall:   ", recall_score(y_true, y_pred, average='macro'))
    print("Macro Avg F1 Score: ", f1_score(y_true, y_pred, average='macro'))

    print("Weighted Avg Precision:", precision_score(y_true, y_pred, average='weighted'))
    print("Weighted Avg Recall:   ", recall_score(y_true, y_pred, average='weighted'))
    print("Weighted Avg F1 Score: ", f1_score(y_true, y_pred, average='weighted'))

# Run classification metric evaluations
for k in [5, 10]:
    evaluate_classification_metrics(eval_df, k)
