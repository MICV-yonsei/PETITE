import pandas as pd
import numpy as np
import openpyxl

def make_new_df(max_epoch, val_every):
    epochs = list(range(val_every-1, max_epoch, val_every))
    cols = []
    for epoch in epochs:
        cols.extend([
            f'{epoch}_192192136', f'{epoch}_192192128', f'{epoch}_22422481', f'{epoch}_12812890', f'{epoch}_12812863',
            ])
    df_partial = pd.DataFrame(
        columns=cols,
        index=[
            'p_192192136', 'p_192192128', 'p_22422481', 'p_12812890', 'p_12812863',
            's_192192136', 's_192192128', 's_22422481', 's_12812890', 's_12812863',
            'n_192192136', 'n_192192128', 'n_22422481', 'n_12812890', 'n_12812863',
            't_192192136', 't_192192128', 't_22422481', 't_12812890', 't_12812863'
            ]
    )

    df_all = {'1': df_partial.copy(), '2': df_partial.copy(), '3': df_partial.copy()}
    return df_all

def fill_df(df_all, psnr, ssim, nrmse, time, epoch, args):
    fold = args.pretrained_model_name.split("#")[1].split(".")[0]
    # base = args.pretrained_model_name.split("#")[0]
    base = args.pretrained_model_name.split("#")[0].split("_")[1]
    target = args.data_dir.split("/")[-2]


    if base == "192":
        source = "192192128"
    elif base == "136":
        source = "192192136"
    elif base == "224":
        source = "22422481"
    elif base == "90":
        source = "12812890"
    elif base == "63":
        source = "12812863"



    df = df_all[fold]
    col = f"{epoch}_{target}"
    p = f"p_{source}"
    s = f"s_{source}"
    n = f"n_{source}"
    t = f"t_{source}"
    df.loc[p, col] = psnr
    df.loc[s, col] = ssim
    df.loc[n, col] = nrmse
    df.loc[t, col] = time

    df_all[fold] = df
    save_df(df_all, args.csv_dir)

    return df_all


def save_df(df, csv_dir):
    with pd.ExcelWriter(csv_dir) as writer:
        for key, value in df.items():
            value.to_excel(writer, sheet_name=key, header=True, index=True)