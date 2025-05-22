import pandas as pd
import matplotlib.pyplot as plt

# Load ROC data
df_esn  = pd.read_csv("roc_data_esn.csv")
df_gru  = pd.read_csv("roc_data_gru.csv")
df_lstm = pd.read_csv("roc_data_lstm.csv")

# Combine all into one DataFrame
df_all = pd.concat([df_esn, df_gru, df_lstm], ignore_index=True)

# Plot
plt.figure(figsize=(7,6))
for model in df_all["model"].unique():
    df_model = df_all[df_all["model"] == model]
    # average over classes
    for class_label in df_model["class"].unique():
        df_sub = df_model[df_model["class"] == class_label]
        plt.plot(df_sub["fpr"], df_sub["tpr"], label=f"{model} ({class_label})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Across Models")
plt.legend()
plt.tight_layout()
plt.show()
