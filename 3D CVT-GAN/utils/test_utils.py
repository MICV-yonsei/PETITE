import pandas as pd
import numpy as np
import openpyxl

def make_new_df():
    psnr = pd.DataFrame(
        columns=['192 192 136', '192 192 128', '224 224 81', '128 128 90', '128 128 63', 'avg'],
        index=['192 192 136', '192 192 128', '224 224 81', '128 128 90', '128 128 63', 'avg']
    )
    ssim = pd.DataFrame(
        columns=['192 192 136', '192 192 128', '224 224 81', '128 128 90', '128 128 63', 'avg'],
        index=['192 192 136', '192 192 128', '224 224 81', '128 128 90', '128 128 63', 'avg']
    )
    nrmse = pd.DataFrame(
        columns=['192 192 136', '192 192 128', '224 224 81', '128 128 90', '128 128 63', 'avg'],
        index=['192 192 136', '192 192 128', '224 224 81', '128 128 90', '128 128 63', 'avg']
    )

    df = {'psnr': psnr, 'ssim': ssim, 'nrmse': nrmse}
    return df 

def fill_df(df, psnr, ssim, nrmse, args):
    resolution_mapping = {'192_136': (1,0), '224_136': (2,0), '90_136': (3,0), '63_136': (4,0),
                        '136_192': (0,1), '224_192': (2,1), '90_192': (3,1), '63_192': (4,1),
                        '136_224': (0,2), '192_224': (1,2), '90_224': (3,2), '63_224': (4,2),
                        '136_90': (0,3), '192_90': (1,3), '224_90': (2,3), '63_90': (4,3),
                        '136_63': (0,4), '192_63': (1,4), '224_63': (2,4), '90_63': (3,4)}
    
    if args.type is not None:
        resolution_key = [key for key in resolution_mapping.keys() if key in args.ckpt_path][0]
    else:
        if '136' in args.data_dir:
            dest = '136'
        elif '192192128' in args.data_dir:
            dest = '192'
        elif '224' in args.data_dir:
            dest = '224'
        elif '90' in args.data_dir:
            dest = '90'
        elif '63' in args.data_dir:
            dest = '63'
        resolution_key = f'{args.base_path.split("/")[-1].split("#")[0]}_{dest}'

    resolution_idx = resolution_mapping[resolution_key]

    df['psnr'].iloc[resolution_idx] = psnr
    df['ssim'].iloc[resolution_idx] = ssim
    df['nrmse'].iloc[resolution_idx] = nrmse

    df['psnr'] = average_df(df['psnr'])
    df['ssim'] = average_df(df['ssim'])
    df['nrmse'] = average_df(df['nrmse'])
    save_df(df, args.csv_dir)
    return df

def average_df(df):
    df.iloc[0,5] = df.iloc[0, [1,2,3,4]].mean()
    df.iloc[1,5] = df.iloc[1, [0,2,3,4]].mean()
    df.iloc[2,5] = df.iloc[2, [0,1,3,4]].mean()
    df.iloc[3,5] = df.iloc[3, [0,1,2,4]].mean()
    df.iloc[4,5] = df.iloc[4, [0,1,2,3]].mean()

    df.iloc[5,0] = df.iloc[[1,2,3,4], 0].mean()
    df.iloc[5,1] = df.iloc[[0,2,3,4], 1].mean()
    df.iloc[5,2] = df.iloc[[0,1,3,4], 2].mean()
    df.iloc[5,3] = df.iloc[[0,1,2,4], 3].mean()
    df.iloc[5,4] = df.iloc[[0,1,2,3], 4].mean()

    df.iloc[5,5] = df.iloc[[0,1,2,3,4], 5].mean()
    return df

def save_df(df, csv_dir):
    with pd.ExcelWriter(csv_dir) as writer:
        for key, value in df.items():
            value.to_excel(writer, sheet_name=key, header=True, index=True)

def make_compare_df(epochs: list):
    psnr = pd.DataFrame(
        columns=epochs,
        index=[
            '192_136', '224_136', '90_136', '63_136', '136_192', '224_192', '90_192', '63_192',
            '136_224', '192_224', '90_224', '63_224', '136_90', '192_90', '224_90', '63_90','136_63', '192_63', '224_63', '90_63', 'avg'
            ]
        )
    ssim = pd.DataFrame(
        columns=epochs,
        index=[
            '192_136', '224_136', '90_136', '63_136', '136_192', '224_192', '90_192', '63_192',
            '136_224', '192_224', '90_224', '63_224', '136_90', '192_90', '224_90', '63_90','136_63', '192_63', '224_63', '90_63', 'avg'
            ]
        )
    nrmse = pd.DataFrame(
        columns=epochs,
        index=[
            '192_136', '224_136', '90_136', '63_136', '136_192', '224_192', '90_192', '63_192',
            '136_224', '192_224', '90_224', '63_224', '136_90', '192_90', '224_90', '63_90','136_63', '192_63', '224_63', '90_63', 'avg'
            ]
        )
    df = {'psnr': psnr, 'ssim': ssim, 'nrmse': nrmse}
    return df

def fill_compare(df, psnr, ssim, nrmse, ckpt_path, csv_dir):
    resolution_mapping = {
        '192_136': 0, '224_136': 1, '90_136': 2, '63_136': 3, '136_192': 4, '224_192': 5, '90_192': 6, '63_192': 7,
        '136_224': 8, '192_224': 9, '90_224': 10, '63_224': 11, '136_90': 12, '192_90': 13, '224_90': 14, '63_90': 15, '136_63': 16, '192_63': 17, '224_63': 18, '90_63': 19
        }              
    resolution_key = [key for key in resolution_mapping.keys() if key in ckpt_path][0]
    epoch = int(ckpt_path.split("_")[-1].split(".")[0])

    resolution_idx = resolution_mapping[resolution_key]
    df['psnr'][epoch].iloc[resolution_idx] = psnr
    df['ssim'][epoch].iloc[resolution_idx] = ssim
    df['nrmse'][epoch].iloc[resolution_idx] = nrmse

    df['psnr'][epoch].iloc[20] = df['psnr'][epoch].iloc[0:20].mean()
    df['ssim'][epoch].iloc[20] = df['ssim'][epoch].iloc[0:20].mean()
    df['nrmse'][epoch].iloc[20] = df['nrmse'][epoch].iloc[0:20].mean()

    save_df(df, csv_dir)
    return df