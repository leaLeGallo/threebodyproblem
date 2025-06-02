import pandas as pd
import matplotlib.pyplot as plt

# File names for CSV inputs
files = {
    'LSTM': 'cumulative_prediction_error_LSTM.csv',
    'GRU':  'cumulative_prediction_error_GRU.csv',
    'ESN':  'cumulative_prediction_error_ESN.csv'
}

# Define font settings for each element
title_font  = {     'size': 25}
label_font  = {   'size': 16}
legend_font = {     'size': 14}
tick_font   = {      'size': 12}

# Read data and plot
plt.figure(figsize=(10, 6))
for model_name, file_path in files.items():
    df = pd.read_csv(file_path)
    plt.plot(
        df['timestep'],
        df['cumulative_mae'],
        label=model_name
    )

# Apply titles and labels with custom fonts
plt.title('Cumulative Prediction Error Comparison', fontdict=title_font)
plt.xlabel('Timestep',                      fontdict=label_font)
plt.ylabel('Cumulative MAE',                fontdict=label_font)

# Customize legend font
plt.legend(prop=legend_font)

# Customize tick labels font
ax = plt.gca()
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(tick_font['size'])

plt.grid(True)
plt.tight_layout()

# Save the plot BEFORE plt.show() to ensure the figure is captured
output_path = 'cumulative_error_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")

# Show the plot on screen
plt.show()