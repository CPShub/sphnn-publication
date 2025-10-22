from . import colors

model_style = {
    'true':     dict(c='black',                          ls='--',  lw=2.0, alpha=1.0),
    'sPHNN':    dict(c=colors.theme_colors['red'],       ls='-',   lw=2.0, alpha=0.8),
    'sPHNN-LM': dict(c=colors.theme_colors['orange'],    ls='-.',  lw=1.0, alpha=0.8),
    'PHNN':     dict(c=colors.theme_colors['lightblue'], ls=':',   lw=1.0, alpha=0.8),
    'sNODE':    dict(c=colors.theme_colors['grey'],      ls='-',   lw=1.0, alpha=0.8),
    'NODE':     dict(c=colors.theme_colors['green'],     ls='-',   lw=1.0, alpha=0.8),
    'cPHNN':    dict(c=colors.theme_colors['darkblue'],  ls=(0, (8, 4, 2, 4, 2, 4)),  lw=1.0, alpha=0.8),
}

model_names = {
    "sPHNN": "sPHNN",
    "sPHNN-LM": "sPHNN-LM",
    "PHNN": "PHNN",
    "NODE": "NODE",
    "sNODE": "sNODE",
    "cPHNN": "bPHNN",
}