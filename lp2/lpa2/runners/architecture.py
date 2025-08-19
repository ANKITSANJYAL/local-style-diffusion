import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# Prompt box
ax.add_patch(Rectangle((0, 2), 1.5, 0.8, facecolor="#dddddd", edgecolor='black'))
ax.text(0.75, 2.4, "Prompt", ha='center', va='center', fontsize=10)

# Parser box
ax.add_patch(Rectangle((2, 2), 1.5, 0.8, facecolor="#f2f2f2", edgecolor='black'))
ax.text(2.75, 2.4, "Parser\n(spaCy/POS)", ha='center', va='center', fontsize=9)

# Token sets
ax.add_patch(Rectangle((4, 3), 1.5, 0.5, facecolor="#b3cde3", edgecolor='black'))
ax.text(4.75, 3.25, r"$T_{obj}$", ha='center', va='center', fontsize=10, fontweight='bold')

ax.add_patch(Rectangle((4, 2), 1.5, 0.5, facecolor="#fbb4ae", edgecolor='black'))
ax.text(4.75, 2.25, r"$T_{style}$", ha='center', va='center', fontsize=10, fontweight='bold')

# U-Net blocks
block_colors = ['#b3cde3', '#b3cde3', '#ccebc5', '#fbb4ae', '#fbb4ae']
block_labels = ['Down 1', 'Down 2', 'Mid', 'Up 1', 'Up 2']

x_start = 6.5
for i, label in enumerate(block_labels):
    ax.add_patch(Rectangle((x_start + i*1.5, 2), 1.2, 1, facecolor=block_colors[i], edgecolor='black'))
    ax.text(x_start + i*1.5 + 0.6, 2.5, label, ha='center', va='center', fontsize=9)

# Generated image box
ax.add_patch(Rectangle((x_start + 5*1.5 + 1, 2), 1.5, 0.8, facecolor="#dddddd", edgecolor='black'))
ax.text(x_start + 5*1.5 + 1.75, 2.4, "Generated\nImage", ha='center', va='center', fontsize=9)

# Arrows: Prompt -> Parser -> Tokens
ax.add_patch(FancyArrowPatch((1.5, 2.4), (2, 2.4), arrowstyle='->', mutation_scale=10))
ax.add_patch(FancyArrowPatch((3.5, 2.4), (4, 3.25), arrowstyle='->', mutation_scale=10))
ax.add_patch(FancyArrowPatch((3.5, 2.4), (4, 2.25), arrowstyle='->', mutation_scale=10))

# Arrows: Tokens into U-Net
# Object tokens to Down blocks
for i in range(2):
    ax.add_patch(FancyArrowPatch((5.5, 3.25), (6.5 + i*1.5, 3.2), arrowstyle='->', mutation_scale=10, color="#2166ac"))

# Style tokens to Mid/Up blocks
for i in range(2, 5):
    ax.add_patch(FancyArrowPatch((5.5, 2.25), (6.5 + i*1.5, 2.2), arrowstyle='->', mutation_scale=10, color="#b2182b"))

# Arrow: U-Net to Generated Image
ax.add_patch(FancyArrowPatch((x_start + 5*1.5, 2.4), (x_start + 5*1.5 + 1, 2.4), arrowstyle='->', mutation_scale=10))

# Legend
ax.add_patch(Rectangle((0, 0.5), 0.4, 0.4, facecolor="#b3cde3", edgecolor='black'))
ax.text(0.5, 0.7, "Object token injection", fontsize=8, va='center')
ax.add_patch(Rectangle((3, 0.5), 0.4, 0.4, facecolor="#fbb4ae", edgecolor='black'))
ax.text(3.5, 0.7, "Style token injection", fontsize=8, va='center')

plt.tight_layout()
plt.show()
