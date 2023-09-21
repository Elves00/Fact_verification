import pandas as pd
file_path = "Master_Claims.csv"
master_claims = pd.read_csv(file_path, sep='|')

print(master_claims)
column_names = master_claims.columns
print(column_names)
claims = []
label = []
evidence_accumulator = []
# Iterate through the rows
for index, row in master_claims.iterrows():
    # Append the claim every 6 rows
    if index % 6 == 0:
        # Combine all pieces of evidence into one long string
        combined_evidence = " ".join(str(evidence) for evidence in evidence_accumulator)

        print(index)

        # Combine claim and evidence pieces
        claim = row['Claim_text'] + " ".join(combined_evidence)

        # Append the claim and combined evidence to their respective lists
        claims.append(claim)
        label.append(row['Label'])

        # Reset the evidence accumulator
        evidence_accumulator = []

    # Append the current evidence to the accumulator
    evidence_accumulator.append(row['Evidence_text'])

# If there are any remaining rows with evidence, add them as the last entry
if evidence_accumulator:
    combined_evidence = " ".join(evidence_accumulator)
    claims.append(master_claims.iloc[-1]['Claim_text'])
    # evidence.append(combined_evidence)
    label.append(row['Label'])


print(len(claims))
# print(len(evidence))
print(len(label))