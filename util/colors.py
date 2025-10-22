
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import colorsys

colors = { 
    'oxford_blue':      { 0: '#021131', 1: '#030a',   2: '#010714', 3: '#010a1d', 4: '#020e27', 5: '#021131', 6: '#062f89', 7: '#094ee1', 8: '#4f84f8', 9: '#a7c2fb' }, 
    'turkey_red':       { 0: '#ad0508', 1: '#230102', 2: '#450203', 3: '#680305', 4: '#8b0407', 5: '#ad0508', 6: '#ed070b', 7: '#f93e41', 8: '#fb7e80', 9: '#fdbfc0' }, 
    'orange_peel':      { 0: '#faa12e', 1: '#3a2101', 2: '#734303', 3: '#ad6404', 4: '#e78506', 5: '#faa12e', 6: '#fbb458', 7: '#fcc782', 8: '#fddaab', 9: '#feecd5' }, 
    'pigment_green':    { 0: '#02a152', 1: '#2010',   2: '#014121', 3: '#016131', 4: '#018141', 5: '#02a152', 6: '#02e674', 7: '#31fd97', 8: '#76feba', 9: '#bafedc' }, 
    'vivid_sky_blue':   { 0: '#14d7f5', 1: '#022c33', 2: '#045966', 3: '#068599', 4: '#08b2cc', 5: '#14d7f5', 6: '#43dff7', 7: '#72e7f9', 8: '#a1effb', 9: '#d0f7fd' }, 
    'electric_indigo':  { 0: '#5835f3', 1: '#0d0338', 2: '#1a0770', 3: '#270aa8', 4: '#340de0', 5: '#5835f3', 6: '#795df6', 7: '#9b86f8', 8: '#bcaefa', 9: '#ded7fd' }, 
    'hollywood_cerise': { 0: '#f519a5', 1: '#340222', 2: '#680443', 3: '#9c0665', 4: '#d987',   5: '#f519a5', 6: '#f747b7', 7: '#f975c9', 8: '#fba3db', 9: '#fdd1ed' }, 
    'antique_white':    { 0: '#fdedde', 1: '#5a2e05', 2: '#b35c0a', 3: '#f38b2a', 4: '#f8bc84', 5: '#fdedde', 6: '#fdf0e4', 7: '#fef4eb', 8: '#fef8f2', 9: '#fffbf8' } }

theme_colors = { 
    'green':        '#16a48a',
    'lightblue':    '#688fc6',
    'darkblue':     '#435384',
    'grey':         '#cccccc',
    'orange':       '#f6a315',
    'red':          '#c24c4c',
    'black':        '#000000',
}

cycle = cycler(color=[
    colors['oxford_blue'][0],
    colors['turkey_red'][6],
    colors['vivid_sky_blue'][2],
    colors['orange_peel'][4],
])

theme_cycle = cycler(color=[theme_colors[k] for k in ['darkblue', 'red', 'lightblue', 'orange', 'green']])


def set_custom_cycle():
    plt.rc('axes', prop_cycle=cycle)

def scale_lightness(hex, scale_l):
    rgb = matplotlib.colors.ColorConverter.to_rgb(hex)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)


set_custom_cycle()

