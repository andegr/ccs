import logging
import re
from pathlib import Path

def create_plots_directory():
    """
    Create a plots directory if it does not already exist and returns it.
    """
    # Define the root output directory
    HERE = Path(__file__).resolve().parent.parent
    PLOTS_DIR = HERE / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)
    logging.info(f'Created plots directory: {PLOTS_DIR}')

    return PLOTS_DIR

def create_output_directory():
    """
    Create an outputs directory with an auto-incremented name based on existing directories.
    The directory will be named 'run_<number>', where <number> is the next available integer.
    """

    # Define the root output directory
    HERE = Path(__file__).resolve().parent.parent
    ROOT_OUT = HERE / "outputs"
    ROOT_OUT.mkdir(exist_ok=True)

    # Find existing run directories and determine the next index
    existing = [p for p in ROOT_OUT.iterdir() if p.is_dir() and p.name.startswith("run_")]
    run_ids = []
    
    for p in existing:
        m = re.match(r"run_(\d+)$", p.name)
        if m:
            run_ids.append(int(m.group(1)))

    run_idx = max(run_ids, default=0) + 1
    OUTDIR = ROOT_OUT / f"run_{run_idx}"
    OUTDIR.mkdir()
    
    print(f"[INFO] Writing all log outputs to {OUTDIR.relative_to(HERE)}")
    return OUTDIR

def setup_logging(OUTDIR):
    # Define a consistent log message format
    FMT = "%(asctime)s  [%(levelname)s]  %(message)s"
    # Configure the root logger
    logging.basicConfig(level=logging.INFO, format=FMT, datefmt="%H:%M:%S")
    fh = logging.FileHandler(OUTDIR / "simulation.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter(FMT, datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)
