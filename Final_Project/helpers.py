def fmt_float(x, max_decimals=3):
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return f"{x:.{max_decimals}f}".rstrip("0").rstrip(".")