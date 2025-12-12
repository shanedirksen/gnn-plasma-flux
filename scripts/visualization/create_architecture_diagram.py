import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
input_color = '#E8F4F8'
embed_color = '#B8E6F0'
message_color = '#7EC8E3'
readout_color = '#4AA8D8'
output_color = '#2E86AB'

# Box styling
box_style = "round,pad=0.1"

# INPUT BOX
input_box = FancyBboxPatch((1.5, 8.5), 7, 0.8, 
                           boxstyle=box_style, 
                           facecolor=input_color, 
                           edgecolor='black', 
                           linewidth=2)
ax.add_patch(input_box)
ax.text(5, 8.9, 'INPUT', ha='center', va='center', 
        fontsize=13, fontweight='bold')
ax.text(5, 8.6, '64 nodes × 4 features', ha='center', va='center', 
        fontsize=10)

# Arrow 1
arrow1 = FancyArrowPatch((5, 8.5), (5, 7.9),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow1)

# EMBEDDING BOX
embed_box = FancyBboxPatch((2, 7.2), 6, 0.7, 
                          boxstyle=box_style, 
                          facecolor=embed_color, 
                          edgecolor='black', 
                          linewidth=2)
ax.add_patch(embed_box)
ax.text(5, 7.6, 'Embedding', ha='center', va='center', 
        fontsize=12, fontweight='bold')
ax.text(5, 7.35, '64 nodes × 128 features', ha='center', va='center', 
        fontsize=9)

# Arrow 2
arrow2 = FancyArrowPatch((5, 7.2), (5, 6.6),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow2)

# MESSAGE PASSING BOX
message_box = FancyBboxPatch((1.5, 5.2), 7, 1.4, 
                            boxstyle=box_style, 
                            facecolor=message_color, 
                            edgecolor='black', 
                            linewidth=2)
ax.add_patch(message_box)
ax.text(5, 6.3, 'Message Passing (×4)', ha='center', va='center', 
        fontsize=12, fontweight='bold')
ax.text(5, 5.9, 'Concat [node(128), neighbors(128)]', ha='center', va='center', 
        fontsize=9)
ax.text(5, 5.55, 'Linear(256 → 128)', ha='center', va='center', 
        fontsize=9)

# Arrow 3
arrow3 = FancyArrowPatch((5, 5.2), (5, 4.6),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow3)

# EDGE READOUT BOX
readout_box = FancyBboxPatch((1.5, 3.5), 7, 1.1, 
                            boxstyle=box_style, 
                            facecolor=readout_color, 
                            edgecolor='black', 
                            linewidth=2)
ax.add_patch(readout_box)
ax.text(5, 4.35, 'Edge Readout', ha='center', va='center', 
        fontsize=12, fontweight='bold')
ax.text(5, 4.0, 'Concat [h_i, h_j]', ha='center', va='center', 
        fontsize=9)
ax.text(5, 3.7, 'Linear(256 → 1)', ha='center', va='center', 
        fontsize=9)

# Arrow 4
arrow4 = FancyArrowPatch((5, 3.5), (5, 2.9),
                        arrowstyle='->', mutation_scale=25, 
                        linewidth=2.5, color='black')
ax.add_patch(arrow4)

# OUTPUT BOX
output_box = FancyBboxPatch((1.5, 2.2), 7, 0.7, 
                           boxstyle=box_style, 
                           facecolor=output_color, 
                           edgecolor='black', 
                           linewidth=2)
ax.add_patch(output_box)
ax.text(5, 2.65, 'OUTPUT', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='white')
ax.text(5, 2.4, '128 edge fluxes', ha='center', va='center', 
        fontsize=10, color='white')

# Add title
ax.text(5, 9.8, 'FluxGNN Architecture', ha='center', va='top', 
        fontsize=15, fontweight='bold')

plt.tight_layout()
plt.savefig('results/gnn_architecture_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Created architecture diagram: results/gnn_architecture_diagram.png")
