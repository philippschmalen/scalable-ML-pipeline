"""
Validate data
* selected columns in config should be in data
* #selected columns should be equal to #columns in data
* feature dtypes according config
"""


def test_columns_in_df(df_census, config):
    """
    Test if selected columns are in data
    """
    columns = (
        [config["data"]["target"]]
        + config["data"]["cat_features"]
        + config["data"]["num_features"]
    )

    for col in columns:
        assert col in df_census.columns, f"{col} not in data"


def test_num_columns(df_census, config):
    """
    Test if selected columns are in data
    """
    columns = (
        [config["data"]["target"]]
        + config["data"]["cat_features"]
        + config["data"]["num_features"]
    )

    assert len(columns) == len(
        df_census.columns
    ), "#columns in config != #columns in data"


def test_feature_dtypes(df_census, config):
    df = df_census[df_census.columns.difference([config["data"]["target"]])]

    assert (
        df.select_dtypes("number")
        .columns.isin(config["data"]["num_features"])
        .all()
    ), "num_features have not numeric type in df"

    assert (
        df.select_dtypes("object")
        .columns.isin(config["data"]["cat_features"])
        .all()
    ), "cat_features have not object type in df"


def test_target_labels(df_census, config):
    """Checks that labels are part of target variable"""

    assert (
        df_census[config["data"]["target"]]
        .isin(config["data"]["target_labels"])
        .all()
    ), "Some of target labels"
