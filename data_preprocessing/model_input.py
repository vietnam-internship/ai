from typing import Iterable, Optional, Tuple
import pandas as pd


def build_model_dataset(
    df: pd.DataFrame,
    target_column: str = "rate",
    feature_columns: Optional[Iterable[str]] = None,
    horizon: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        return df.loc[:, []], pd.Series(dtype=float)

    if feature_columns is None:
        feature_columns = [
            c for c in df.columns if c not in ("date", "source", "cur_unit", target_column)
        ]
    feature_columns = list(feature_columns)

    target_col = f"{target_column}_target"
    shifted_groups = []
    for _, group in df.groupby(["source", "cur_unit"], sort=False):
        group = group.sort_values("date").copy()
        group[target_col] = group[target_column].shift(-horizon)
        shifted_groups.append(group)

    result = pd.concat(shifted_groups).sort_index()
    result = result.dropna(subset=[*feature_columns, target_col])

    X = result[feature_columns].reset_index(drop=True)
    y = result[target_col].reset_index(drop=True)
    return X, y
