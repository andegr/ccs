def fmt_float(x, max_decimals=3):
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")

def build_traj_fname(
    prefix,
    n_particles,
    t_sim,
    dt,
    v0,
    Dt,
    Dr,
    run_id,
    walls = False,
    L = 0,
    ext=".txt",
):
    
    wall_tag = "_walls" if walls else "" 
    L_tag = f"_L{fmt_float(L)}" if walls else ""

    return (
        f"{prefix}{wall_tag}{L_tag}"
        f"_n{n_particles}"
        f"_tsim{t_sim}"
        f"_dt{dt}"
        f"_v0{fmt_float(v0)}"
        f"_Dt{fmt_float(Dt)}"
        f"_Dr{fmt_float(Dr)}"
        f"_run{fmt_float(run_id)}{ext}"
    )  
