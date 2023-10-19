from sklearn.metrics import cohen_kappa_score
import numpy as np
from scipy.stats import norm

# Labels assigned by Rebecca and Brecon
rebecca_labels = ['T', 'F', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'F', 'T', 'N', 'T', 'F', 'F', 'N', 'N', 'N', 'N', 'T', 'T', 'T', 'F', 'T', 'N', 'T', 'F', 'F', 'T', 'F']
brecon_labels = ['T', 'F', 'T', 'T', 'N', 'T', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'T', 'F', 'T', 'T', 'F', 'T', 'N', 'T', 'F', 'F', 'N', 'N', 'N', 'N', 'T', 'T', 'T', 'F', 'T', 'N', 'T', 'F', 'F', 'T', 'F']

cohen_kappa = cohen_kappa_score(rebecca_labels, brecon_labels)
print("Cohen's Kappa:", cohen_kappa)

agreement = sum(1 for a, b in zip(rebecca_labels, brecon_labels) if a == b) / len(rebecca_labels)

unique_labels = set(rebecca_labels)
pe = sum((rebecca_labels.count(label) / len(rebecca_labels)) * (brecon_labels.count(label) / len(rebecca_labels)) for label in unique_labels)

kappa = (agreement - pe) / (1 - pe)

# Calculate the standard error (SE) of Cohen's Kappa
n = len(rebecca_labels)
SE = np.sqrt((agreement * (1 - pe)) / n)
print(SE)
# Set the desired confidence level (e.g., 95%)
confidence_level = 0.95

# Calculate the z-score for the desired confidence level
z_score = norm.ppf((1 + confidence_level) / 2)
print(z_score)
# Calculate the margin of error
margin_of_error = z_score * SE

# Calculate the confidence interval
lower_bound = kappa - margin_of_error
upper_bound = kappa + margin_of_error

# Print the results
print("Cohen's Kappa:", kappa)
print(f"Confidence Interval ({int(confidence_level * 100)}%): [{lower_bound}, {upper_bound}]")