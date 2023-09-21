from transformers import DataCollatorWithPadding,Trainer, AutoTokenizer, TFTrainingArguments, TFTrainer, TFBertModel, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
import pandas as pd
from datasets import Dataset
import evaluate
import numpy as np
from huggingface_hub import notebook_login
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import TFAutoModelForSequenceClassification

from transformers import create_optimizer
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback

from transformers.keras_callbacks import PushToHubCallback


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

accuracy = evaluate.load("accuracy")



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "FALSE", 1: "NOT_ENOUGH_INFO", 2: "TRUE"}
label2id = {"FALSE": 0, "NOT_ENOUGH_INFO": 1, "TRUE": 2}

# model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id, token=access_token
)

training_args = TrainingArguments(
    output_dir="training_model",
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


batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)


model = TFAutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id
)

tf_train_set = model.prepare_tf_dataset(
    tokenized_dataset["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_dataset["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)


model.compile(optimizer=optimizer)  # No loss argument!

metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)

push_to_hub_callback = PushToHubCallback(
    output_dir="validation_model",
    tokenizer=tokenizer,
)

callbacks = [metric_callback, push_to_hub_callback]

model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3, callbacks=callbacks)