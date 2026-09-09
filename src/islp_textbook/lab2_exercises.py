"""
Marimo notebook for Lab 2 Exercises from the ISLP textbook.
Run with `uv run marimo edit src/islp_textbook/lab2_exercises.py` to view the
notebook in a web browser.
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import pandas as pd
    import plotly.express as px


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Lab 2 Exercises

    A set of exercises from Chapter 2 of the ISLP textbook.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conceptual
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 1

    For each of parts (a) through (d), indicate whether we would generally
    expect the performance of a flexible statistical learning method to be
    better or worse than an inflexible method. Justify your answer.
    (a) The sample size n is extremely large, and the number of predictors p is small.
    (b) The number of predictors p is extremely large, and the number
    of observations n is small.
    (c) The relationship between the predictors and response is highly
    non-linear.
    (d) The variance of the error terms, i.e. σ2 = Var("), is extremely
    high.

    a) Here we have a large number of observations, so we can be relatively less concerned with overfitting, for the same number of training cycles, preferring a flexible model. High bias, low variance.

    b) The reverse of the above, so we prefer an inflexible model. In this scenario, we have low bias and high variance, therefore simple, inflexible model is better. We guard against overfitting.

    c) Non-linear relationships are harder to capture with inflexible models like parametarised ones which cannot capture the non-linear nature. So we prefer flexible.

    d) High variance therefore we prefer (as in b) an inflexible model.

    ### 2

    Explain whether each scenario is a classification or regression problem, and indicate whether we are most interested in inference or prediction. Finally, provide n and p.
    (a) We collect a set of data on the top 500 firms in the US. For each
    firm we record profit, number of employees, industry and the
    CEO salary. We are interested in understanding which factors
    affect CEO salary.
    (b) We are considering launching a new product and wish to know
    whether it will be a success or a failure. We collect data on 20
    similar products that were previously launched. For each product we have recorded whether it was a success or failure, price
    charged for the product, marketing budget, competition price,
    and ten other variables.
    (c) We are interested in predicting the % change in the USD/Euro
    exchange rate in relation to the weekly changes in the world
    stock markets. Hence we collect weekly data for all of 2012. For
    each week we record the % change in the USD/Euro, the %
    change in the US market, the % change in the British market,
    and the % change in the German market.

    a) Inference because we care about which factors, not what the salary is. Regression because salary is continuous output variable. n=500. p=3.

    b) Prediction because we care about success or failure, value of the output bool is of upmost importance, not why. Classification because output is a bool. n=20. p=13.
    c) Prediction. Regression. n=52. p=3.

    ### 3

    We now revisit the bias-variance decomposition.
    (a) Provide a sketch of typical (squared) bias, variance, training error, test error, and Bayes (or irreducible) error curves, on a single plot, as we go from less flexible statistical learning methods
    towards more flexible approaches. The x-axis should represent
    the amount of flexibility in the method, and the y-axis should
    represent the values for each curve. There should be five curves.
    Make sure to label each one.
    (b) Explain why each of the five curves has the shape displayed in
    part (a).

    a) ![alt](public/sketch1.png)

    b) skip

    ### 4

    You will now think of some real-life applications for statistical learning.
    (a) Describe three real-life applications in which classification might
    be useful. Describe the response, as well as the predictors. Is the
    goal of each application inference or prediction? Explain your
    answer.
    (b) Describe three real-life applications in which regression might
    be useful. Describe the response, as well as the predictors. Is the
    goal of each application inference or prediction? Explain your
    answer.
    (c) Describe three real-life applications in which cluster analysis
    might be useful.

    a) skip

    b) skip

    c) Species of plant based on phenotypic characteristics measured, when unclear what species are present. Finding the most "anomalous" transactions in a dataset of transactions which could contain fraudulent transactions. Persona analysis in marketing.

    ### 5 - skip

    ### 6

    Describe the differences between a parametric and a non-parametric
    statistical learning approach. What are the advantages of a parametric approach to regression or classification (as opposed to a nonparametric approach)? What are its disadvantages?

    A parametric approach has linear, real numbers as parameters in a supposed function, and the goal is to find the parameters such that the function best approximates the dataset, with minimal error. A non-parametric approach, like deep learning, does not suppose any particular functional form. The advantages of a parametric approach are lower risk of overfitting, and higher computational efficiency. The disadvantages are a lack of flexibility (unable to model the true relationship) and high bias towards the measured input parameters.

    ### 7

    The table below provides a training data set containing six observations, three predictors, and one qualitative response variable.
    Obs. X1 X2 X3 Y
    1 0 3 0 Red
    2 2 0 0 Red
    3 0 1 3 Red
    4 0 1 2 Green
    5 −1 0 1 Green
    6 1 1 1 Red
    Suppose we wish to use this data set to make a prediction for Y when
    X1 = X2 = X3 =0 using K-nearest neighbors.
    (a) Compute the Euclidean distance between each observation and
    the test point, X1 = X2 = X3 =0.
    (b) What is our prediction with K =1? Why?
    (c) What is our prediction with K =3? Why?
    (d) If the Bayes decision boundary in this problem is highly nonlinear, then would we expect the best value for K to be large or
    small? Why?

    a) each is the square root of the sum of the 3 values squared. 1: 3. 2: 2. 3: 3.162. 4: 2.236. 5: 1.414. 6: 1.732.
    b) Prediction is green, because 5 is the closest other point.
    c) Prediction is red, because 2, 5 and 6 are the 3 closest points in space, and they are more red than green.
    d) Small. At higher K, more points are averaged out, and so the classification dividing lines are closer to linear.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Applied
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 8

    College.csv
    """)
    return


@app.cell
def _():
    # a
    college = pd.read_csv("local_assets/College.csv")

    # b
    pd.read_csv("local_assets/College.csv", index_col=0)
    college3 = college.rename({"Unnamed: 0": "College"}, axis=1)
    college3 = college3.set_index("College")

    college = college3
    college3
    return (college,)


@app.cell
def _(college):
    # c
    college.describe()
    return


@app.cell
def _(college):
    # d

    first_columns_only_df = college[["Top10perc", "Apps", "Enroll"]]

    fig = px.scatter_matrix(first_columns_only_df)
    fig
    return


@app.cell
def _(college):
    # e
    college.boxplot(column="Outstate", by="Private")
    return


@app.cell
def _(college):
    # f
    college["Elite"] = pd.cut(college["Top10perc"], [0, 0.5, 1], labels=["No", "Yes"])
    college["Elite"].value_counts()
    return


@app.cell
def _(college):
    college.boxplot("Outstate", "Elite")
    return


@app.cell
def _(college):
    # Do private schools get more applications or less? ie. does the average private
    # college need to sift through more or less applications?
    private_vs_apps = college.groupby("Private")["Apps"].mean()
    private_vs_apps.plot(kind="bar", x="Private", y="Apps")
    return


@app.cell
def _(college):
    college.boxplot(column="Apps", by="Private")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Less!
    """)
    return


if __name__ == "__main__":
    app.run()
