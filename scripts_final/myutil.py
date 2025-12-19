import numpy as np
import pandas as pd
import re, io
import matplotlib.pyplot as plt

def float_extract(mylist):
    float_list = []
    for i in mylist.split(' '):
        if len(i):
            digits = re.findall(r'[-+]?\d*\.?\d+(?:[e][-+]?\d+)?', i)
            if len(digits):
                float_list.append(float(digits[0]))
    return np.array(float_list)


def convert_to_csv(input_fname, output_fname):
    #used in case the df is not saves
    # Read the text file
    with open(input_fname, 'r') as file:
        lines = file.readlines()

    # Define a pattern to extract the relevant data
    pattern = r"\[(\d+), ([\d\.]+), array\(\[(.*?)\]\)\]"

    data = []

    for line in lines:
        match = re.search(pattern, line)
        if match:
            row_id = int(match.group(1))
            value = float(match.group(2))
            array_values = list(map(float, match.group(3).split(',')))
            data.append([row_id, value] + array_values)

    # Convert the data into a DataFrame
    df = pd.DataFrame(data, columns=['subID', 'nll', 'alpha', 'beta', 'gamma', 'theta'])
    # Save the DataFrame as a CSV file
    df.to_csv(output_fname, index=True)

#global parameter
def get_attack_prob(x, max_attack_prob):
    """
    get the predator's attack probability given the location that the players choose
    """
    if x>2:
        return (x/20)**2*max_attack_prob
    else:
        return 0
    #return max(0, max_attack_prob - (steps - x)*0.1)

def get_potential_reward(x):
    """
    get potential reward of a location
    """
    return x**2

def get_predator_choice(x, ptype):
    """
    determine if predator attack given the location that the players choose
    """
    if ptype==1:
        prob = get_attack_prob(x, max_attack_prob=4.8)
    elif ptype==0:
        prob = get_attack_prob(x, max_attack_prob=2.6)
    c = np.random.choice([0, 1], 1, p=[1-prob, prob])
    return c



def results_to_df(res, num_betas):
    """Accepts either a statsmodels results."""
    conf = res.conf_int()
    df = pd.DataFrame({
        "coef": res.params,
        "std err": res.bse,
        "t": res.tvalues,
        "P>|t|": res.pvalues,
        "[0.025, 0.975]": list(zip(conf[0].round(3), conf[1].round(3))),
    })
    df = df.iloc[0: num_betas]
    return df

def get_sig(p):
    if p<1e-4:
        sig = "****"
    elif p<1e-3:
        sig = "***"
    elif p<1e-2:
        sig = "**"
    elif p<0.05:
        sig = "*"
    elif p<0.06:
        sig = '.'
    else:
        sig = "n.s."
    return sig

def add_sigstars(p, cut=(0.05, 0.01, 0.001)):
    # † for p<0.001 (you can swap if you prefer)
    # if p < cut[3]: return '†'
    if p < cut[2]: return '***'
    if p < cut[1]: return '**'
    if p < cut[0]: return '*'
    return ''

def format_df_for_print(df):
    out = df.copy()
    # Try to detect column names across SM variants
    pcol = next((c for c in out.columns if re.search(r'p[>|]\|?t?\|?', c, re.I)), None)
    if pcol is None:
        pcol = next((c for c in out.columns if c.lower().startswith("p")), None)

    # Round numbers nicely
    for c in out.columns:
        if np.issubdtype(out[c].dtype, np.number):
            out[c] = out[c].map(lambda x: f"{x:.3f}")

    # Append significance stars to coef if p available
    if pcol and "coef" in out.columns:
        out["coef"] = [
            f"{coef}{add_sigstars(float(p))}"
            for coef, p in zip(out["coef"], df[pcol])
        ]
    return out

def df_to_png(df, formula, title, folder, dpi=300, font_size=10):
    fig, ax = plt.subplots(figsize=(10, 0.5+0.4*len(df.index))) #0.5 + 0.4*len(df.index))
    filename = f"{title}_{folder}.png"
    ax.axis('off')
    # Title
    ax.set_title(title+"\n", fontsize=font_size+2, pad=12, x=0.4) #pad=12
    # Make table
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        cellLoc='right',
        colLoc='right',
        loc='center'
    )
    # add formula
    if formula is not None:
        # Put formula as a separate line just below the title
        plt.text(0.35, 1.05, formula, #(0.35, 1.05)
                 ha='center', va='bottom',
                 fontsize=font_size, transform=ax.transAxes)

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    tbl.scale(0.75, 2)  # a little taller and narrower rows
    fig.tight_layout()

    #save
    fig.savefig(f"../paper_figs/{folder}/{filename}", dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close(fig)



