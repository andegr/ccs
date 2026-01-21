def fmt_float(x, max_decimals=3):
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")

def build_traj_fname(
    prefix,
    n_particles,
    t_sim,
    t_eq,
    dt,
    L,
    v0,
    Dt,
    Dr,
    run_id,
    walls = False,
    pairwise = False,
    eta = 0,            # area fraction, only included if pairwise=True
    ext=".txt",
    disc = False,
    r_disc = 0,
    epsilon_disc = 0,
):
    
    wall_tag = "_walls" if walls else "" 
    pairwise_tag = "_pw" if pairwise else "" 
    eta_tag = f"_eta{fmt_float(eta, max_decimals=2)}" if pairwise else "" # only plotted if pairwise=True
    L_tag = f"_L{fmt_float(L, max_decimals=1)}" if (walls or pairwise) else "" # only plot L together with walls and pairwise
    disc_tag = f"_disc_rdisc{fmt_float(r_disc, max_decimals=1)}_epsilondisc{fmt_float(epsilon_disc, max_decimals=1)}" if disc else ""

    return (
        f"{prefix}{wall_tag}{pairwise_tag}{disc_tag}"
        f"_n{n_particles}"
        f"_tsim{t_sim}"
        f"_teq{t_eq}"
        f"_dt{dt}"
        f"{L_tag}"
        f"_v0{fmt_float(v0)}"
        f"_Dt{fmt_float(Dt)}"
        f"_Dr{fmt_float(Dr)}"
        f"{eta_tag}"
        f"_run{fmt_float(run_id)}{ext}"
    )  


