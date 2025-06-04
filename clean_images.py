import os
import matplotlib.pyplot as plt
from PIL import Image

# Create output directory
output_dir = "cleaned_results"
os.makedirs(output_dir, exist_ok=True)

# Define the alpha values and opponent types
alphas = ["0.2", "0.4", "0.6", "0.8"]
opponent_types = ["aggressive", "loose", "passive", "tight"]

for opponent in opponent_types:
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    # fig.suptitle(f'{opponent.capitalize()} Opponents', fontsize=20)

    for i, alpha in enumerate(alphas):
        bankroll_path = f"results/{opponent}_opponents_{alpha}_bankroll_alpha_{alpha}.png"
        kl_div_path = f"results/{opponent}_opponents_{alpha}_kl_div_alpha_{alpha}.png"

        # Load and display bankroll image
        if os.path.exists(bankroll_path):
            bankroll_img = Image.open(bankroll_path)
            axs[0, i].imshow(bankroll_img)
            axs[0, i].set_title(f'Bankroll (α={alpha})')
        else:
            axs[0, i].text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=14)

        # Load and display KL divergence image
        if os.path.exists(kl_div_path):
            kl_img = Image.open(kl_div_path)
            axs[1, i].imshow(kl_img)
            axs[1, i].set_title(f'KL Divergence (α={alpha})')
        else:
            axs[1, i].text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=14)

        axs[0, i].axis('off')
        axs[1, i].axis('off')

    # Save the figure
    output_path = os.path.join(output_dir, f"{opponent}_opponents_combined.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

types = [
    ("ultra_aggressive", "Ultra-Aggressive"),
    ("uniform", "Uniform")
]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i, (prefix, label) in enumerate(types):
    bankroll_path = f"results/{prefix}_opponents_bankroll_alpha_NA.png"
    kl_div_path = f"results/{prefix}_opponents_kl_div_alpha_NA.png"

    # Bankroll (top row)
    if os.path.exists(bankroll_path):
        img = Image.open(bankroll_path)
        axs[0, i].imshow(img)
        axs[0, i].set_title(f"{label} Bankroll", fontsize=10)
    else:
        axs[0, i].text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=12)
    axs[0, i].axis('off')

    # KL Divergence (bottom row)
    if os.path.exists(kl_div_path):
        img = Image.open(kl_div_path)
        axs[1, i].imshow(img)
        axs[1, i].set_title(f"{label} KL Divergence", fontsize=10)
    else:
        axs[1, i].text(0.5, 0.5, 'Missing', ha='center', va='center', fontsize=12)
    axs[1, i].axis('off')

# Save the combined image
output_path = os.path.join(output_dir, "special_opponents_combined.png")
plt.tight_layout(pad=0.5)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
