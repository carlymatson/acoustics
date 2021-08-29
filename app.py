import streamlit as st
import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns
import base64
from datetime import date
from math import isclose

file1 = "9x9 assigining pv full quad 07272021.xlsx"
file2 = "9x9 spot check in pv angles 07302021.xlsx"
file3 = "9x9 inverted cosine 08052021.xlsx"
file4 = "9x9 inverted sine  08052021.xlsx"

st.set_page_config(layout = "wide")
sns.set_style("whitegrid")


def download_widget(object_to_download, download_file = "download.csv", key = None):
    col1, col2 = st.beta_columns(2)
    col1.write("Table shape (rows x columns):")
    col1.write(object_to_download.shape)
    filename = col2.text_input("Give a name to the download file", download_file, key = key)
    my_link = download_link(object_to_download, filename, "Click to Download")
    col2.markdown(my_link, unsafe_allow_html = True)


def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def select_grid(modulus = 3):
    with st.beta_expander("Grid Points"):
        m_bounds = st.slider("X bounds", value = (-4,4), min_value = -10, max_value = 10)
        n_bounds = st.slider("Y bounds", value = (-4,4), min_value = -10, max_value = 10)
        mn_list = [(m,n) for m in range(m_bounds[0], m_bounds[1]+1) \
                         for n in range(n_bounds[0], n_bounds[1]+1)]
        df = pd.DataFrame({"points": mn_list})
        df[["x", "y"]] = df["points"].to_list()
        df["points"] = df["points"].astype(str)
        st.write(df)
        sns.scatterplot(data = df, x = "x", y = "y")
        st.write(plt.gcf())
    return mn_list

def select_pointing_vecs(modulus = 3):
    with st.beta_expander("Pointing Vectors"):
        num_vecs = st.slider("Number of vectors", value = 84, min_value = 1, max_value = 100)
        pq_list = evenly_spaced_vectors(num = num_vecs)
        placeholder = st.empty()
        on_axis = [v for v in pq_list if isclose(v[0], 0, abs_tol=1e-5) or isclose(v[1],0, abs_tol=1e-5)]
        exclude = st.multiselect(
            "Exclude:", 
            options = pq_list,
            default = on_axis,
            format_func = lambda v: "(%0.3f, %0.3f)"%(v[0], v[1])
        )
        include_origin = st.checkbox("Include the zero vector", value = True)
        pq_list_edited = list(pq_list)
        for v in exclude:
            pq_list_edited.remove(v)
        if include_origin:
            pq_list_edited.insert(0, (0,0))
        df = pd.DataFrame({"vectors": pq_list_edited})
        df[["p", "q"]] = df["vectors"].to_list()
        df["vectors"] = df["vectors"].astype(str)
        num = len(pq_list_edited)
        st.write(df)
        fig = ff.create_quiver(
            [0]*num, 
            [0]*num, 
            *list(zip(*pq_list_edited)), 
            scale = 1, 
            scaleratio = 1,
            arrow_scale = 0.07
        )
        placeholder.write(fig)
    return pq_list_edited

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
    df["mn"] = df["mn"].astype(str)
    df["pq"] = df["pq"].astype(str)
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

# FIXME 9x9 is hard-coded.
def main():
    #modulus = st.slider("Modulus", min_value = 2, max_value = 10, value = 3)
    modulus = 3
    st.header("Select points and vectors")
    mn_list = select_grid(modulus = modulus)
    pq_list = select_pointing_vecs(modulus = modulus)

    st.header("Computations")
    df = compute_matrices(mn_list, pq_list, modulus)
    st.write(df)
    download_widget(
        df, 
        key="computations_download", 
        download_file = "9x9_computations_" + str(date.today()) + ".csv"
    )
    # Plot
    plot_zetas = False
    if plot_zetas:
        fig, ax = plt.subplots()
        sns.scatterplot(data = df, x = "cos", y = "sin", ax = ax)
        st.write(fig)

    cos_df = pd.pivot(data = df, index = "pq", columns = "mn", values = "cos")
    sin_df = pd.pivot(data = df, index = "pq", columns = "mn", values = "sin")

    st.header("Complex matrix")
    complex_mat = np.matrix(cos_df) + np.matrix(sin_df)*1j
    complex_df = pd.DataFrame(complex_mat).applymap(lambda z: "%0.3f + %0.3fi"%(z.real, z.imag)) 
    complex_df.index = cos_df.index
    complex_df.columns = cos_df.columns
    st.write(complex_df)
    det = np.linalg.det(complex_mat)
    st.write("Determinant:", det)
    download_widget(
        complex_df, 
        key="complex_download",
        download_file = "9x9_g_%s.csv"%(str(date.today()))
    )



if __name__ == "__main__":
    main()