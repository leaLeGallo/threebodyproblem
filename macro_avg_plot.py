import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and combine your ROC data
df_esn  = pd.read_csv("roc_data_esn.csv")
df_gru  = pd.read_csv("roc_data_gru.csv")
df_lstm = pd.read_csv("roc_data_lstm.csv")
df_all = pd.concat([df_esn, df_gru, df_lstm], ignore_index=True)

# Common FPR grid for interpolation
mean_fpr = np.linspace(0, 1, 200)

plt.figure(figsize=(7, 6))

for model in df_all["model"].unique():
    df_model = df_all[df_all["model"] == model]
    tprs = []
    for cls in df_model["class"].unique():
        sub = df_model[df_model["class"] == cls]
        interp_tpr = np.interp(mean_fpr, sub["fpr"], sub["tpr"])
        interp_tpr[0], interp_tpr[-1] = 0.0, 1.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    auc_macro = np.trapz(mean_tpr, mean_fpr)
    plt.plot(mean_fpr, mean_tpr,
             label=f"{model} (AUC={auc_macro:.2f})",
             linewidth=2)

# Diagonal chance line
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1)

# Only set fontsize here; default font and weight remain
plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)
plt.title("Macro-averaged ROC by Model", fontsize=20)
plt.legend()
plt.tight_layout()

plt.savefig("roc_macro_avg.png", dpi=300)
plt.show()
