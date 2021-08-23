import streamlit as st
import pandas as pd
import math
import numpy as np

file1 = "9x9 assigining pv full quad 07272021.xlsx"
file2 = "9x9 spot check in pv angles 07302021.xlsx"
file3 = "9x9 inverted cosine 08052021.xlsx"
file4 = "9x9 inverted sine  08052021.xlsx"


def compute_dft(df, modulus, p = "p", q = "q", m = "m", n = "n"): # also modulus
    # assert that the columns x, y, m, n are in df.
    df["dot"] = df[p] * df[m] + df[q] * df[n]
    df["angle"] = df["dot"] * 2 / modulus
    df["cos"] = np.cos(df["angle"] * math.pi)
    df["sin"] = np.sin(df["angle"] * math.pi)
    return df


def main():
    st.write("Hello, Gary!")

    df = pd.read_excel(file1)
    st.write(df)

    modulus = 3
    x_list = [i for i in range(modulus)]
    y_list = list(x_list)
    p_list = list(x_list)
    q_list = list(x_list)
    index = pd.MultiIndex.from_product(
        [x_list, y_list, p_list, q_list], 
        names = ["m", "n", "p", "q"]
    )

    new_df = pd.DataFrame(index = index).reset_index()
    new_df = compute_dft(new_df, modulus = 3)
    st.write(new_df)
    st.write(new_df.shape)


if __name__ == "__main__":
    main()