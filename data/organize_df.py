# average the results for 3 folds
import pandas as pd
import numpy as np
import argparse
import re

parser = argparse.ArgumentParser(description="Organize result dataframe")
parser.add_argument("--load_dir", default="", type=str)

def new_sheet(args):
    df = load_df(args)
    
    avg_1 = average_fold(df['1'], fold=1)
    avg_2 = average_fold(df['2'], fold=2)
    avg_3 = average_fold(df['3'], fold=3)
    avg = pd.concat([avg_1, avg_2, avg_3], axis=0)
    df['avg'] = average_sheet(avg)

    save_dir = f'{args.load_dir.split(".")[0]}_organized.xlsx'
    save_df(df, save_dir)
    print("Finish")

def find_minmax(df):
    cols = list(df.columns)
    prev = df.loc[:,cols].apply(pd.to_numeric)
    df['max'] = prev.max(axis=1)
    df['max_row'] = prev.idxmax(axis=1)
    df['min'] = prev.min(axis=1)
    df['min_row'] = prev.idxmin(axis=1)
    
    return df

def average_sheet(df):
    cols = list(df.columns)
    avg_df = pd.DataFrame(columns=cols, index=['psnr','ssim','nrmse','time'])
    
    for col in cols:
        avg_df.loc['psnr', col] = df.loc[['p_#1', 'p_#2', 'p_#3'],col].mean()
        avg_df.loc['ssim', col] = df.loc[['s_#1', 's_#2', 's_#3'],col].mean()
        avg_df.loc['nrmse', col] = df.loc[['n_#1', 'n_#2', 'n_#3'],col].mean()
        avg_df.loc['time', col] = df.loc[['t_#1', 't_#2', 't_#3'],col].mean()

    df = pd.concat([df, avg_df], axis=0)
    df = find_minmax(df)

    return df

def average_fold(df, fold):
    cols = list(df.columns)
    rows = [f'{m}_#{fold}' for m in ['p','s','n','t']]
    epochs = sorted(list(set([int(re.match(r'(\d+)_', col).group(1)) for col in cols])))
    resolutions = ['192192136', '192192128', '22422481', '12812890', '12812863']

    avg_psnr = average_metric(df, 'p', resolutions, epochs)
    avg_ssim = average_metric(df, 's', resolutions, epochs)
    avg_nrmse = average_metric(df, 'n', resolutions, epochs)
    avg_time = average_metric(df, 't', resolutions, epochs)
    
    avg = pd.DataFrame([avg_psnr, avg_ssim, avg_nrmse, avg_time], columns=epochs, index=rows)
    return avg

def average_metric(df, metric, resolutions, epochs):
    avgs = []
    for epoch in epochs:
        ep_resolution = [f'{epoch}_{res}' for res in resolutions]
        me_resolution = [f'{metric}_{res}' for res in resolutions]
        subset = df.loc[me_resolution, ep_resolution]
        avgs.append(average_df(subset))

    return avgs

def average_df(df):
    val = df.values.flatten()
    val = [x for x in val if not pd.isna(x)]
    return np.mean(val)

def load_df(args):
    df = {
            '1': pd.read_excel(args.load_dir, sheet_name='1', header=0, index_col=0),
            '2': pd.read_excel(args.load_dir, sheet_name='2', header=0, index_col=0),
            '3': pd.read_excel(args.load_dir, sheet_name='3', header=0, index_col=0),
            }    
    
    return df

def save_df(df, csv_dir):
    with pd.ExcelWriter(csv_dir) as writer:
        for key, value in df.items():
            value.to_excel(writer, sheet_name=key, header=True, index=True)


if __name__=="__main__":
    args = parser.parse_args()
    new_sheet(args)