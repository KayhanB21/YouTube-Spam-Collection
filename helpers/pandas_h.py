# coding=utf-8
# Kayhan B
# V0.1
# Jun 2022
# derived Helper (can import other helpers)
import pandas as pd
from IPython.display import display


def null_val_summary(df: pd.DataFrame) -> None:
    null_info = pd.DataFrame(df.dtypes).T.rename(index={0: 'column type'})
    null_info = pd.concat([null_info, pd.DataFrame(df.isnull().sum()).T.rename(index={0: 'null values (nb)'})])
    null_info = pd.concat(
        [null_info, pd.DataFrame(df.isnull().sum() / len(df.index) * 100).T.rename(index={0: 'null values (%)'})])
    display(null_info)


def unique_col_percent(df: pd.DataFrame) -> None:
    for col in df.columns:
        print(
            f"{col} unique count and percentage: {len(df[col].unique())}, {len(df[col].unique()) / len(df.index) * 100:0.2f}%")
