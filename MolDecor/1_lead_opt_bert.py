import pandas as pd
from transformers import DataCollatorWithPadding
from datasets import Dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


np.random.seed(1)

checkpoint = 'unikei/bert-base-smiles'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# replace the dataset with full csv file (Can be acquired from corresponding author)
df = pd.read_csv('data/freq_600_sample_dataset.csv')
df.sample(frac=1, random_state=42)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    df.Scaffold, df.label, 
    test_size=0.2,       # 10% of data for validation
    stratify=df.label,          # Stratify by the class labels
    random_state=42      # Set random seed for reproducibility
)
train_df = df.iloc[y_train.index.values.tolist()]
train_df = train_df.reset_index().iloc[:, 1:]
eval_df = df.iloc[y_val.index.values.tolist()]
eval_df = eval_df.reset_index().iloc[:, 1:]


labels_count = len(set(train_df.label.values.tolist()))
print('labels_count:', labels_count)

# Create a preprocessing function to tokenize text and truncate sequences to be no longer than BERT-BASE maximum input length:
def preprocess_function(examples):
    return tokenizer(examples["Scaffold"], truncation=True)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(train_df.label.values.ravel().tolist()), y=train_df.label.values.ravel().tolist())

pd.DataFrame({'weights':class_weights}).to_csv('data/class_weights_600.csv')

class_weights =  torch.tensor(class_weights.tolist(), dtype=torch.float32)
class_weights = class_weights.to('cuda:0')

print(class_weights)

# Loading the custom dataset into dataset class
train_ds = Dataset.from_dict(train_df.to_dict(orient='list'))
eval_ds = Dataset.from_dict(eval_df.to_dict(orient='list'))

train_tokenized_scaffolds = train_ds.map(preprocess_function, batched=True)
eval_tokenized_scaffolds = eval_ds.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=labels_count)

training_args = TrainingArguments(
    output_dir="/data/checkpoints/",
    learning_rate=2e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    disable_tqdm=True, # to stop the progress bar.
    gradient_accumulation_steps=10,
    save_total_limit=3
)

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

# # Example custom loss function
def custom_loss_fn(logits, labels):
    loss_fct = torch.nn.CrossEntropyLoss(class_weights)
    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

# # Initialize your Trainer with the CustomTrainer
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


