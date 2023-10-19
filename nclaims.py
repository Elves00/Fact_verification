from transformers import DataCollatorWithPadding,Trainer, AutoTokenizer
import pandas as pd
from datasets import Dataset
import evaluate
import numpy as np
from huggingface_hub import notebook_login
from transformers import create_optimizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification, TrainingArguments, Trainer
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback



def dataframe_from_excel(filepath,sep=None):
    claims = []
    label = []
    evidence_accumulator = []
    
    if sep == None:
        # Load claims
        raw_fact_ver_dataframe = pd.read_excel(filepath)
    else:
        raw_fact_ver_dataframe = pd.read_csv(filepath,sep=sep)


    # Iterate through the rows
    for index, row in raw_fact_ver_dataframe.iterrows():
        # Append the claim every 6 rows
        if index % 6 == 0:
            # Combine all pieces of evidence into one long string
            combined_evidence = "".join(str(evidence) for evidence in evidence_accumulator)

            # Combine claim and evidence pieces
            claim = row[2] + "".join(combined_evidence)

            # Append the claim and combined evidence to their respective lists
            claims.append(claim)
            label.append(row[3])

            # Reset the evidence accumulator
            evidence_accumulator = []

        # Append the current evidence to the accumulator
        evidence_accumulator.append(row[4])

    # If there are any remaining rows with evidence, add them as the last entry
    if evidence_accumulator:
        combined_evidence = "".join(evidence_accumulator)
        claims.append(raw_fact_ver_dataframe.iloc[-1][2])
        # evidence.append(combined_evidence)
        label.append(row[3])
    
    # Create a DataFrame
    fact_ver_dataframe = pd.DataFrame({
        'label': label,
        'text': claims
    })
    
    # Create a mapping from labels to numerical values
    label_mapping = {'F': 0, 'N': 1, 'T': 2}

    # Replace the labels in the 'Label' column with numerical values
    fact_ver_dataframe['label'] = fact_ver_dataframe['label'].map(label_mapping)

    return fact_ver_dataframe


dataset = dataframe_from_excel('N Claims.xlsx')

print(dataset)

# Create a Dataset from pandas DataFrame
dataset = Dataset.from_pandas(dataset)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
dataset.push_to_hub("neutral_claim")