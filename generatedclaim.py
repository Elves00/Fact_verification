import pandas as pd
from datasets import Dataset


def dataframe_from_excel(filepath):
    claims = []
    label = []
    evidence_accumulator = []

    # Load claims
    raw_fact_ver_dataframe = pd.read_excel(filepath)

    # Iterate through the rows
    for index, row in raw_fact_ver_dataframe.iterrows():
        print(index)
        # Append the claim every 6 rows
        if index % 6 == 0:
            # Combine all pieces of evidence into one long string
            combined_evidence = "".join(str(evidence) for evidence in evidence_accumulator)

            # Combine claim and evidence pieces
            claim = row['Claim'] + "".join(combined_evidence)

            # Append the claim and combined evidence to their respective lists
            claims.append(claim)
            label.append(row['Overall'])

            # Reset the evidence accumulator
            evidence_accumulator = []

        # Append the current evidence to the accumulator
        evidence_accumulator.append(row['Evidence'])

    # If there are any remaining rows with evidence, add them as the last entry
    if evidence_accumulator:
        combined_evidence = "".join(evidence_accumulator)
        claims.append(raw_fact_ver_dataframe.iloc[-1]['Claim'])
        # evidence.append(combined_evidence)
        label.append(row['Overall'])
    
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


data = dataframe_from_excel("auto_claim_evidence.xlsx")
# Create a Dataset from pandas DataFrame
dataset = Dataset.from_pandas(data)
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
dataset.push_to_hub("Auto_Set")