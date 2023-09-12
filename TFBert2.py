from transformers import DataCollatorWithPadding,Trainer, AutoTokenizer, BertForSequenceClassification, TFTrainingArguments, TFTrainer, TFBertModel, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset

from datasets import load_dataset

from huggingface_hub import notebook_login

access_token = "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"
notebook_login()

claims = []
label = []
evidence_accumulator = []

# Load claims
raw_fact_ver_dataframe = pd.read_excel('C:/Users/breco/Documents/Code/Fact_verification/Claims.xlsx')

# Iterate through the rows
for index, row in raw_fact_ver_dataframe.iterrows():
    # Append the claim every 6 rows
    if index % 6 == 0:
        # Combine all pieces of evidence into one long string
        combined_evidence = " ".join(evidence_accumulator)

        # Combine claim and evidence pieces
        claim = row['Claim'] + " ".join(combined_evidence)

        # Append the claim and combined evidence to their respective lists
        claims.append(claim)
        label.append(row['Overall Label'])

        # Reset the evidence accumulator
        evidence_accumulator = []

    # Append the current evidence to the accumulator
    evidence_accumulator.append(row['Evidence'])

# If there are any remaining rows with evidence, add them as the last entry
if evidence_accumulator:
    combined_evidence = " ".join(evidence_accumulator)
    claims.append(raw_fact_ver_dataframe.iloc[-1]['Claim'])
    # evidence.append(combined_evidence)
    label.append(row['Overall Label'])


print(len(claims))
# print(len(evidence))
print(len(label))

# Create a DataFrame
fact_ver_dataframe = pd.DataFrame({
    'label': label,
    'text': claims
})

# Create a mapping from labels to numerical values
label_mapping = {'F': 0, 'N': 1, 'T': 2}

# Replace the labels in the 'Label' column with numerical values
fact_ver_dataframe['label'] = fact_ver_dataframe['label'].map(label_mapping)

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", tokenizer_options={"truncation": True , "padding": True})

# Create a Pandas DataFrame
dataset = Dataset.from_pandas(fact_ver_dataframe)


print('here')

dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

print(dataset["test"][0])

# Tokenization function
def preprocess_function(token):
    return tokenizer(token["text"], truncation=True)


tokenized_dataset = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")


import evaluate

accuracy = evaluate.load("accuracy")

import numpy as np


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "FALSE", 1: "NOT_ENOUGH_INFO", 2: "TRUE"}
label2id = {"FALSE": 0, "NOT_ENOUGH_INFO": 1, "TRUE": 2}

# model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id, token=access_token
)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.push_to_hub()
