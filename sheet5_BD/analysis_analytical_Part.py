#%%
import numpy as np
import matplotlib.pyplot as plt
from SaveToFile import load_trajectory, load_positions_txt
import Plot
from parameters import A,B,C, kB, T, V_0, n_particles, n_steps_acorr
import os
import time
import logging
from autocorrelation import autocorrelation

Plot.apply_style()

#%%
# --------------- ANALYTICAL PART 1 ---------------- # 

# d) (iii)

def Pot_DW_asym(x_arr):
    return V_0 * (A*x_arr**4 - B * x_arr**2 + C*x_arr)


old_well_position = np.sqrt(B/(2*A))
x_arr = np.linspace(-1.5*old_well_position, 1.5*old_well_position, 1000)

V_DW_asym = Pot_DW_asym(x_arr=x_arr)

extremas_x = np.array([-np.sqrt(B/(2*A)) - C/(4*B), -C/(4*B), np.sqrt(B/(2*A)) - C/(4*B)])

# 4. Calculate the Potential V at these positions
extremas_V = Pot_DW_asym(extremas_x)

# Calculate the Energy Differences (Barrier Height relative to wells)
# Note: Barrier is index 1
Delta_V_Left = extremas_V[1] - extremas_V[0] 
Delta_V_Right = extremas_V[1] - extremas_V[2]


# --- Printing Results ---
labels = ["Left Well", "Barrier Top", "Right Well"]

print(f"{'Location':<15} | {'Position (x)':<15} | {'Potential (V)':<15}")
print("-" * 50)
for label, x, v in zip(labels, extremas_x, extremas_V):
    print(f"{label:<15} | {x:<15.4f} | {v:<15.4f}")


# --- PLOTTING ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(x_arr, V_DW_asym, label=r"$V_\text{DW}(x)$" + "\nfor " + r"$C=0.1/\sigma$", color="black")
ax.axvline(extremas_x[0], -2.3, 1.55, linestyle="dashed", label=r"$\tilde{x}_{3} = - \sqrt{\frac{B}{2A}} -\frac{C}{4B}$")
ax.axvline(extremas_x[1], -2.3, 1.55, linestyle="dashdot", label=r"$\tilde{x}_{1} = - \frac{C}{4B}$")
ax.axvline(extremas_x[2], -2.3, 1.55, linestyle="dotted", label=r"$\tilde{x}_{2} = + \sqrt{\frac{B}{2A}} -\frac{C}{4B}$")
ax.scatter(extremas_x, extremas_V, color='red', zorder=5, )
# --- Add Potential Value at Extremas Labels to the Plot ---
for x, v in zip(extremas_x, extremas_V):
    # Format the label string (e.g., 3 decimal places)
    label_text = f"V(x)={v:.3f}"
    
    # Annotate puts text at a specific coordinate (x, v)
    plt.annotate(
        label_text, 
        (x, v),
        textcoords="offset points", # Use relative offset
        xytext=(10, -15),             # Shift text 15 points vertically up
        ha='center',                # Center the text horizontally
        fontweight='bold',
        # Optional: Add a box behind text for readability
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9)
    )
# ----- VISUALIZING THE DELTAS -----
# 1. Draw a horizontal reference line at the Barrier Height
# Spanning from Left Well to Right Well

ax.hlines(y=extremas_V[0], xmin=extremas_x[0], xmax=extremas_x[2], 
          colors='green', linestyles=':', alpha=0.8)
ax.hlines(y=extremas_V[1], xmin=extremas_x[0], xmax=extremas_x[2], 
          colors='green', linestyles=':', alpha=0.8, label='Barrier Level')
ax.hlines(y=extremas_V[2], xmin=extremas_x[0], xmax=extremas_x[2], 
          colors='green', linestyles=':', alpha=0.8)

# 2. Draw the Vertical Arrows using annotate
# Arrow for Left Well Depth
ax.annotate(
    text="", 
    xy=(extremas_x[0] + 0.33*(extremas_x[1] - extremas_x[0]) , extremas_V[1]),      # Arrow Tip (Barrier level)
    xytext=(extremas_x[0] + 0.33*(extremas_x[1] - extremas_x[0]), extremas_V[0]),  # Arrow Tail (Well bottom)
    arrowprops=dict(arrowstyle='<->', color='green', lw=1.5)
)
# Label for Left Arrow (offset slightly to the left)
ax.text(extremas_x[0] + 0.33*(extremas_x[1] - extremas_x[0]) + 0.66, (extremas_V[0] + extremas_V[1])/2, 
        r"$\Delta V_{{13}} = {:.2f}$".format(Delta_V_Left), color='green', va='center', ha='right', fontweight='bold')

# Arrow for Right Well Depth
ax.annotate(
    text="", 
    xy=(extremas_x[2] - 0.33*(extremas_x[2] - extremas_x[1]) , extremas_V[1]),      # Arrow Tip (Barrier level)
    xytext=(extremas_x[2] - 0.33*(extremas_x[2] - extremas_x[1]) , extremas_V[2]),  # Arrow Tail (Well bottom)
    arrowprops=dict(arrowstyle='<->', color='green', lw=1.5)
)
# Label for Right Arrow (offset slightly to the right)
ax.text(extremas_x[2] - 0.33*(extremas_x[2] - extremas_x[1]) + 0.1, (extremas_V[2] + extremas_V[1])/2, 
        r"$\Delta V_{{12}} = {:.2f}$".format(Delta_V_Right), color='green', va='center', ha='left', fontweight='bold')
# ----- VISUALIZING THE DELTAS -----

# ----- Other Plotting Stuff -----
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$V_\text{DW}(x)$")
# ax.set_xlim(left=-2.75, right=6.5)
# bbox_to_anchor places the legend relative to the axes. (1.05, 1) is just outside the top-right.
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
plt.tight_layout() # Adjust layout to ensure nothing is clipped
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(os.path.dirname(__file__), "plots/Part_1_d_iii_C0.1.pdf"))
# plt.show()




