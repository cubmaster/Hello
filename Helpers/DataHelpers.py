import pandas as pd
from sklearn.preprocessing import Imputer


from matplotlib import pyplot as plt

def get_csv(file_path):
    df = pd.read_csv(file_path)
    return df


def get_sql(sql, conn_server):

    cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                          "Server=localhost;"
                          "Database=DataWarehouse;"
                          "Trusted_Connection=yes;")

    # Excute Query here
    df = pd.read_sql(sql, cnxn)
    return df


def simple_null_check(df):
    return df.isnull().values.any()


def show_null_by_column(df):
    for column_name in df.columns:
        print("# rows with zero in {0} => {1}".format(column_name, len(df.loc[df[column_name] == 0])))


def correlation(df, graph):
    cols = len(df.columns)
    corr = df.corr()
    if graph:
        fig, ax = plt.subplots(figsize=(cols, cols))
        ax.matshow(corr)
        plt.xticks(range(cols), corr.columns)
        plt.yticks(range(cols), corr.columns)
        plt.show()
    return corr


def mean_imputer(missing_value,data):
    fill = Imputer(missing_value, strategy="mean", axis=0)
    return fill.fit_transform(data)


if __name__ == '__main__':
    get_csv('')

