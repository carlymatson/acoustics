import streamlit as st
import pandas as pd
import math
import numpy as np
import seaborn as sns
from datetime import date

from utils import *

sns.set_style("whitegrid")

# A note on notation: 'df' is always short for 'dataframe', a table in the Pandas library.


def compute_dot_and_trig_functions(df, modulus, p="p", q="q", m="m", n="n"):
    """Excel-style computations for a Discrete Fourier Transform.
    The cos and sin columns give the real and imaginary parts of the
    roots of unity appearing in the final matrix."""
    df["dot"] = df[p] * df[m] + df[q] * df[n]
    df["dot * 2pi/9"] = df["dot"] * 2 * math.pi / modulus
    df["cos"] = np.cos(df["dot * 2pi/9"])
    df["sin"] = np.sin(df["dot * 2pi/9"])
    return df


def get_table_of_computations(point_list, vec_list, modulus):
    index = pd.MultiIndex.from_product([point_list, vec_list], names=["mn", "pq"])
    df = pd.DataFrame(index=index).reset_index()
    df[["m", "n"]] = df["mn"].to_list()
    df[["p", "q"]] = df["pq"].to_list()
    df = compute_dot_and_trig_functions(df, modulus=modulus)
    df["mn"] = df["mn"].astype(str)
    df["pq"] = df["pq"].astype(str)
    return df


def show_excel_style_computations(point_list, vector_list, modulus):
    df = get_table_of_computations(point_list, vector_list, modulus)

    st.header("Computations")
    st.write(df)
    download_widget(
        df,
        key="computations_download",
        download_file="9x9_computations_" + str(date.today()) + ".csv",
    )
    return df


def show_complex_matrix(df):
    ### Discrete Fourier Transform ###
    st.header("Complex matrix")

    # Reshape the cosine and sine columns into matrices and combine into complex matrix.
    cos_df = pd.pivot(data=df, index="pq", columns="mn", values="cos")
    sin_df = pd.pivot(data=df, index="pq", columns="mn", values="sin")
    complex_mat = np.matrix(cos_df) + np.matrix(sin_df) * 1j

    # Reduce number of decimal places for readability and set row & column names.
    complex_df = pd.DataFrame(complex_mat).applymap(
        lambda z: "%0.3f + %0.3fi" % (z.real, z.imag)
    )
    complex_df.index = cos_df.index
    complex_df.columns = cos_df.columns

    # Display the matrix and its determinant.
    st.write(complex_df)
    det = np.linalg.det(complex_mat)
    st.write("Determinant:", det)

    # Download the matrix to a CSV file.
    download_widget(
        complex_df,
        key="complex_download",
        download_file="9x9_g_%s.csv" % (str(date.today())),
    )
    return complex_df


def main():
    ### Setting up the acoustic field ###
    st.header("Select points and vectors")

    modulus = 3
    point_list = select_grid(modulus=modulus)
    vector_list = select_pointing_vecs(modulus=modulus)

    ## Display excel-sheet style computations
    df = show_excel_style_computations(point_list, vector_list, modulus)

    ## Show the resulting complex matrix and its determinant
    show_complex_matrix(df)


if __name__ == "__main__":
    main()
