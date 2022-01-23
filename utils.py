import math
import numpy as np
import base64
from typing import Tuple, List
import streamlit as st
import pandas as pd
from math import isclose
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import figure_factory as ff


def points_of_grid(
    x_bounds: Tuple[int, int], y_bounds: Tuple[int, int]
) -> List[Tuple[int, int]]:
    return [(x, y) for x in range(*x_bounds) for y in range(*y_bounds)]


def get_hermitian_matrix(matrix):
    """Takes the transpose and the complex conjugate of each entry."""
    nrows, ncols = matrix.shape
    conjugate_matrix = np.matrix(
        [[matrix[row, col].conjugate() for col in range(ncols)] for row in range(nrows)]
    )
    return conjugate_matrix.transpose()


def evenly_spaced_vectors(number_of_vectors=9):
    """Get rotationally-symmetric vectors pointing outward from the origin."""
    angle = 2 * math.pi / number_of_vectors
    angle_list = [i * angle for i in range(number_of_vectors)]
    vectors = [(np.cos(angle), np.sin(angle)) for angle in angle_list]
    return vectors


def select_grid(modulus=3):
    """Interactive widget to select the grid of points to consider."""
    with st.expander("Grid Points"):
        m_bounds = st.slider("X bounds", value=(-4, 4), min_value=-10, max_value=10)
        n_bounds = st.slider("Y bounds", value=(-4, 4), min_value=-10, max_value=10)
        mn_list = [
            (m, n)
            for m in range(m_bounds[0], m_bounds[1] + 1)
            for n in range(n_bounds[0], n_bounds[1] + 1)
        ]
        df = pd.DataFrame({"points": mn_list})
        df[["x", "y"]] = df["points"].to_list()
        df["points"] = df["points"].astype(str)
        st.write(df)
        sns.scatterplot(data=df, x="x", y="y")
        st.write(plt.gcf())
    return mn_list


def select_pointing_vecs(modulus=3):
    """Interactive widget to select the list of vectors defining the waves in the acoustic field."""
    with st.expander("Pointing Vectors"):
        num_vecs = st.slider("Number of vectors", value=84, min_value=1, max_value=100)
        pq_list = evenly_spaced_vectors(number_of_vectors=num_vecs)
        placeholder = st.empty()
        on_axis = [
            v
            for v in pq_list
            if isclose(v[0], 0, abs_tol=1e-5) or isclose(v[1], 0, abs_tol=1e-5)
        ]
        exclude = st.multiselect(
            "Exclude:",
            options=pq_list,
            default=on_axis,
            format_func=lambda v: "(%0.3f, %0.3f)" % (v[0], v[1]),
        )
        include_origin = st.checkbox("Include the zero vector", value=True)
        pq_list_edited = list(pq_list)
        for v in exclude:
            pq_list_edited.remove(v)
        if include_origin:
            pq_list_edited.insert(0, (0, 0))
        df = pd.DataFrame({"vectors": pq_list_edited})
        df[["p", "q"]] = df["vectors"].to_list()
        df["vectors"] = df["vectors"].astype(str)
        num = len(pq_list_edited)
        st.write(df)
        fig = ff.create_quiver(
            [0] * num,
            [0] * num,
            *list(zip(*pq_list_edited)),
            scale=1,
            scaleratio=1,
            arrow_scale=0.07,
        )
        placeholder.write(fig)
    return pq_list_edited


def download_link(object_to_download, download_filename, download_link_text):
    """Encodes an object into a downloadable link."""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def download_widget(object_to_download, download_file="download.csv", key=None):
    """Interactive widget to name a CSV file for download."""
    col1, col2 = st.columns(2)
    col1.write("Table shape (rows x columns):")
    col1.write(object_to_download.shape)
    filename = col2.text_input(
        "Give a name to the download file", download_file, key=key
    )
    my_link = download_link(object_to_download, filename, "Click to Download")
    col2.markdown(my_link, unsafe_allow_html=True)
