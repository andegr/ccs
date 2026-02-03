import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from SaveToFile import load_cluster_results
from Plot import set_Plot_Font, apply_style
import os

set_Plot_Font()
apply_style()
os.chdir(os.path.dirname(__file__))

plots_path = os.path.join(os.path.dirname(__file__), "plots")



#-----------load data--------------------

(
    smax_vals,
    nclus_vals,
    nmono_vals,
    smeanw_vals,
    v0_arr,
    n_particles_arr,
    r_cut_list,
) = load_cluster_results("cluster_results_nsweep.txt")




# ---------- plotting only 3 observables ----------
colors = {
    n_particles: f"C{i}"
    for i, n_particles in enumerate(n_particles_arr)
}


linestyles = {
    r_cut_list[0]: "--",
    r_cut_list[1]: "-",
    r_cut_list[2]: "--",
}

alphas = {
    r_cut_list[0]: 0.3,
    r_cut_list[1]: 1,
    r_cut_list[2]: 0.3,
}


labels_rcut = {
    r_cut_list[0]: rf"${r_cut_list[0]:.2f}\,r_c$",
    r_cut_list[1]: rf"${r_cut_list[1]:.2f}\,r_c$",
    r_cut_list[2]: rf"${r_cut_list[2]:.2f}\,r_c$",
}

fig, axes = plt.subplots(3, 1, figsize=(10,10),sharex=True)

for n_particles in n_particles_arr:
    for r_cut in r_cut_list:
        y = [smax_vals[v0, n_particles, r_cut] for v0 in v0_arr]
        axes[0].plot(
            v0_arr, y,
            color=colors[n_particles],
            linestyle=linestyles[r_cut],
            alpha = alphas[r_cut], 
            marker="o",
            # label=f"N={n_particles}, {labels_rcut[r_cut]}"
        )

axes[0].set_ylabel(r"$\langle s_{\max} \rangle$")
# axes[0].legend(fontsize=8)
axes[0].grid(True)


# (2) Number of clusters <n_clus>
for n_particles in n_particles_arr:
    for r_cut in r_cut_list:
        y = [nclus_vals[v0, n_particles, r_cut] for v0 in v0_arr]
        axes[1].plot(
            v0_arr, y/n_particles,
            color=colors[n_particles],
            linestyle=linestyles[r_cut],
            alpha = alphas[r_cut],
            marker="o"
        )

axes[1].set_ylabel(r"$\langle n_{\mathrm{clus}} / n \rangle$")
axes[1].grid(True)


# # (3) Number of monomers <n_mono>
# for n_particles in n_particles_arr:
#     for r_cut in r_cut_list:
#         y = [nmono_vals[v0, n_particles, r_cut] for v0 in v0_arr]
#         axes[2].plot(
#             v0_arr, y/n_particles,
#             color=colors[n_particles],
#             linestyle=linestyles[r_cut],
#             alpha = alphas[r_cut],
#             marker="o"
#         )

# axes[2].set_ylabel(r"$\langle n_{\mathrm{mono}} / n \rangle$")
# axes[2].grid(True)


# (4) Weighted mean cluster size <s_mean^w>
for n_particles in n_particles_arr:
    for r_cut in r_cut_list:
        y = [smeanw_vals[v0, n_particles, r_cut] for v0 in v0_arr]
        axes[2].plot(
            v0_arr, y,
            color=colors[n_particles],
            linestyle=linestyles[r_cut],
            alpha = alphas[r_cut],
            marker="o"
        )

axes[2].set_ylabel(r"$\langle s_{\mathrm{mmw}} \rangle$")
axes[2].set_xlabel(r"$v_0 \, \sigma / \tau_{{BD}}$")
axes[2].grid(True)

# legends

handles_N = [
    Line2D(
        [], [],
        color=colors[n],
        linestyle="-",
        marker="o",
        label=rf"$N = {n}$",
    )
    for n in n_particles_arr
]



leg1 = axes[0].legend(
    handles=handles_N,
    title=rf"$r_{{\mathrm{{cluster}}}}=(1\,\pm0.01)\,r_{{cut}}$",
    #fontsize=8,
    title_fontsize=17,
    loc="center left", bbox_to_anchor=(1.02, -0.5),
    #bbox_to_anchor=(1.02, 1.0),
    frameon=False,
)


axes[0].add_artist(leg1)



plt.tight_layout(rect=[0, 0, 0.7, 1])
plt.savefig(os.path.join(plots_path, "cluster_analysis.png"), dpi=400)





#-----all observables-------------

# colors = {
#     n_particles: f"C{i}"
#     for i, n_particles in enumerate(n_particles_arr)
# }


# linestyles = {
#     r_cut_list[0]: "--",
#     r_cut_list[1]: "-",
#     r_cut_list[2]: "--",
# }

# alphas = {
#     r_cut_list[0]: 0.3,
#     r_cut_list[1]: 1,
#     r_cut_list[2]: 0.3,
# }


# labels_rcut = {
#     r_cut_list[0]: rf"${r_cut_list[0]:.2f}\,r_c$",
#     r_cut_list[1]: rf"${r_cut_list[1]:.2f}\,r_c$",
#     r_cut_list[2]: rf"${r_cut_list[2]:.2f}\,r_c$",
# }

# fig, axes = plt.subplots(4, 1, figsize=(10,10),sharex=True)

# for n_particles in n_particles_arr:
#     for r_cut in r_cut_list:
#         y = [smax_vals[v0, n_particles, r_cut] for v0 in v0_arr]
#         axes[0].plot(
#             v0_arr, y,
#             color=colors[n_particles],
#             linestyle=linestyles[r_cut],
#             alpha = alphas[r_cut], 
#             marker="o",
#             # label=f"N={n_particles}, {labels_rcut[r_cut]}"
#         )

# axes[0].set_ylabel(r"$\langle s_{\max} \rangle$")
# # axes[0].legend(fontsize=8)
# axes[0].grid(True)


# # (2) Number of clusters <n_clus>
# for n_particles in n_particles_arr:
#     for r_cut in r_cut_list:
#         y = [nclus_vals[v0, n_particles, r_cut] for v0 in v0_arr]
#         axes[1].plot(
#             v0_arr, y/n_particles,
#             color=colors[n_particles],
#             linestyle=linestyles[r_cut],
#             alpha = alphas[r_cut],
#             marker="o"
#         )

# axes[1].set_ylabel(r"$\langle n_{\mathrm{clus}} / n \rangle$")
# axes[1].grid(True)


# # (3) Number of monomers <n_mono>
# for n_particles in n_particles_arr:
#     for r_cut in r_cut_list:
#         y = [nmono_vals[v0, n_particles, r_cut] for v0 in v0_arr]
#         axes[2].plot(
#             v0_arr, y/n_particles,
#             color=colors[n_particles],
#             linestyle=linestyles[r_cut],
#             alpha = alphas[r_cut],
#             marker="o"
#         )

# axes[2].set_ylabel(r"$\langle n_{\mathrm{mono}} / n \rangle$")
# axes[2].grid(True)


# # (4) Weighted mean cluster size <s_mean^w>
# for n_particles in n_particles_arr:
#     for r_cut in r_cut_list:
#         y = [smeanw_vals[v0, n_particles, r_cut] for v0 in v0_arr]
#         axes[3].plot(
#             v0_arr, y,
#             color=colors[n_particles],
#             linestyle=linestyles[r_cut],
#             alpha = alphas[r_cut],
#             marker="o"
#         )

# axes[3].set_ylabel(r"$\langle s_{\mathrm{mean}}^{(w)} \rangle$")
# axes[3].set_xlabel(r"$v_0 \, \sigma / \tau_{{BD}}$")
# axes[3].grid(True)

# # legends

# handles_N = [
#     Line2D(
#         [], [],
#         color=colors[n],
#         linestyle="-",
#         marker="o",
#         label=rf"$N = {n}$",
#     )
#     for n in n_particles_arr
# ]
# handles_rcut = [
#     Line2D(
#         [], [],
#         color="black",
#         linestyle=linestyles[r],
#         alpha=alphas[r],
#         label=rf"$r_{{cluster}}=$ {labels_rcut[r]}",
#     )
#     for r in r_cut_list
# ]


# leg1 = axes[0].legend(
#     handles=handles_N,
#     title=rf"$r_{{\mathrm{{cluster}}}}=(1\,\pm0.01)\,r_{{cut}}$",
#     #fontsize=8,
#     title_fontsize=17,
#     loc="upper left",
#     #bbox_to_anchor=(1.02, 1.0),
#     frameon=False,
# )


# axes[0].add_artist(leg1)

# # axes[0].legend(
# #     handles=handles_rcut,
# #     title=r"$r_{\mathrm{cluster}}$",
# #     #fontsize=8,
# #     #title_fontsize=9,
# #     loc="upper left",
# #     bbox_to_anchor=(1.02, 0.55),
# #     frameon=False,
# # )






# plt.tight_layout()
# plt.savefig(os.path.join(plots_path, "cluster_analysis.png"), dpi=400)