# Project
This repository contains computations for Gary Koopmann's project on an impedance algorithm based on digital acoustic space.

## Setting up a development environment
All of the code in this project is written in Python 3 and is run using the
Python library Streamlit, which displays an interactive app in your browser. 
Streamlit has a lot of dependencies, so it is a good idea 
to install it inside of a virtual environment.


**Step 1:** Create a virtual environment.

    python3 -m venv myvirtualenv

**Step 2:** Activate your new virtual environment.

    source myvirtualenv/bin/activate

To deactivate the virtual environment, simply enter `deactivate`.

**Step 3:** Install required packages.

    pip install --upgrade pip
    pip install -r requirements.txt


## Further reading and resources

This project touches on many advanced mathematical topics, including the following.

Linear algebra: 
* Vectors
* Matrices
* Dot products
* Determinants
* Matrix inverses
* Matrix transpose and the Hermitian matrix

Complex numbers: 
* Complex roots of unity
* Complex conjugation and the Hermitian matrix
* The Discrete Fourier Transform (DFT)

Wikipedia is a good place to get a high-level overview and find further resources on any of these topics.