import streamlit as st
import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns

file1 = "9x9 assigining pv full quad 07272021.xlsx"
file2 = "9x9 spot check in pv angles 07302021.xlsx"
file3 = "9x9 inverted cosine 08052021.xlsx"
file4 = "9x9 inverted sine  08052021.xlsx"

def select_grid():
    modulus = 3
    with st.beta_expander("Choose grid points"):
        st.write("Choose me!")
        mn_list = [(m,n) for m in range(modulus) for n in range(modulus)]
        df = pd.DataFrame({"points": mn_list})
        df[["x", "y"]] = df["points"].to_list()
        st.write(df)
        sns.scatterplot(data = df, x = "x", y = "y")
        st.write(plt.gcf())
    return mn_list

def select_pointing_vecs():
    with st.beta_expander("Choose pointing vectors"):
        st.write("I point!")
        pq_list = evenly_spaced_vectors()
        fig = ff.create_quiver([0]*9, [0]*9, *list(zip(*pq_list)))
        st.write(fig)
    return pq_list

def compute_dft(df, modulus, p = "p", q = "q", m = "m", n = "n"): # also modulus
    # assert that the columns m, y, m, n are in df.
    df["dot"] = df[p] * df[m] + df[q] * df[n]
    df["angle"] = df["dot"] * 2 / modulus
    df["cos"] = np.cos(df["angle"] * math.pi)
    df["sin"] = np.sin(df["angle"] * math.pi)
    return df

def compute_matrices(point_list, vec_list, modulus):
    index = pd.MultiIndex.from_product(
        [point_list, vec_list],
        names = ["mn", "pq"]
    )
    df = pd.DataFrame(index = index).reset_index()
    df[["m", "n"]] = df["mn"].to_list()
    df[["p", "q"]] = df["pq"].to_list()
    df = compute_dft(df, modulus = modulus)
    return df

def evenly_spaced_vectors(num = 9):
    angle = 2 * math.pi / num
    angle_list = [i*angle for i in range(num)]
    cos_list = np.cos(angle_list)
    sin_list = np.sin(angle_list)
    return list(zip(cos_list, sin_list))

def check_sin_and_cos_matrices():
    gary_sin_df = pd.read_excel("9x9 inverted sine  08052021.xlsx", header=None)
    sin_diagonal = pd.Series([gary_sin_df.iloc[i,i] for i in range(81)], name="sin")
    sin_diagonal_inv = 1/sin_diagonal
    gary_cos_df = pd.read_excel("9x9 inverted cosine 08052021.xlsx", header=None)
    cos_diagonal = pd.Series([gary_cos_df.iloc[i,i] for i in range(81)], name="cos")
    cos_diagonal_inv = 1/cos_diagonal
    circle_df = pd.DataFrame({"cos": cos_diagonal_inv, "sin": sin_diagonal_inv})
    circle_df["angle"] = np.arcsin(circle_df["sin"]) * 42 / math.pi
    st.write(circle_df)
    fig, ax = plt.subplots()
    sns.scatterplot(data = circle_df, x = "cos", y = "sin", ax = ax)
    st.write(fig)

def main():
    sns.set_style("whitegrid")
    with st.beta_expander("Check matrices"):
        check_sin_and_cos_matrices()

    modulus = 3
    mn_list = select_grid()
    pq_list = select_pointing_vecs()
    df = compute_matrices(mn_list, pq_list, modulus)
    st.write(df)
    st.write(df.shape)
    # Plot
    fig, ax = plt.subplots()
    sns.scatterplot(data = df, x = "cos", y = "sin", ax = ax)
    st.write(fig)

    cos_df = pd.pivot(data = df, index = "pq", columns = "mn", values = "cos")
    sin_df = pd.pivot(data = df, index = "pq", columns = "mn", values = "sin")
    st.write("Cosine matrix")
    st.write(cos_df)
    st.write("Sine matrix")
    st.write(sin_df)



if __name__ == "__main__":
    main()