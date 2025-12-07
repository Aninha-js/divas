import numpy as np
import pandas as pd
from typing import Tuple

def fetch_data(source: str) -> Tuple[np.ndarray, np.ndarray]:

    # Lê o CSV
    df = pd.read_csv(source)

    # Extrai o ano da data de lançamento
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    # Seleciona colunas relevantes
    df = df[["genre", "developer", "publisher", "release_year", "copies_sold"]]

    # One-Hot Encoding nas colunas de texto
    df = pd.get_dummies(df, columns=["genre", "developer", "publisher"], drop_first=True)

    # X = todas menos a última (features)
    X = df.drop(columns=["copies_sold"]).values

    # y = última (target)
    y = df["copies_sold"].values

    return X, y
