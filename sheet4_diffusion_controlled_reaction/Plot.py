
import matplotlib.pyplot as plt
import numpy as np


#--------------------- Define Plot and Label Sizes for Latex integration---------------------#

def set_size(width,  scale = 1, fraction=1, square_plot=False):
    """Set figure dimensions to avoid scaling in LaTeX.
    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts

    scale: float, optional
            Additional scaling of figure height to deviate from golden ratio

    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    if square_plot:
        fig_height_in = fig_width_in
    else:
        fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, scale*fig_height_in)
    # print(fig_dim)

    return fig_dim

def set_Plot_Font():

    global universal_font_size, tex_width, savepath
    universal_font_size = 11
    # tex_width = 425.1968503937 #for current document, textwidth
    tex_width = 451.6875
    # width = 448.13095
    # savepath = "/path/to/folder/"

    global tex_fonts
    tex_fonts = {
        # Use LaTeX to write all text (for matching fonts) (disable
        # for plots in presentations since its a font with serifs)
        "text.usetex": False,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": universal_font_size,
        "font.size": universal_font_size-1, #title font size
        # Make the legend/label fonts a little smaller
        "legend.fontsize": universal_font_size-2,
        "xtick.labelsize": universal_font_size-2,
        "ytick.labelsize": universal_font_size-2,
        "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    }

    plt.rcParams.update(tex_fonts)


#--------------------- Plot Functions ---------------------#




