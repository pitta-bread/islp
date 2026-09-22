import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from ISLP import load_data
    from ISLP.models import ModelSpec as MS
    from ISLP.models import poly, summarize
    from matplotlib.pyplot import subplots
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF


@app.cell
def _():
    dir()
    return


@app.cell
def _():
    A = np.array([3, 5, 11])
    dir(A)
    return (A,)


@app.cell
def _(A):
    A.sum()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simple Linear Regression
    """)
    return


@app.cell
def _():
    Boston = load_data("Boston")
    Boston.columns
    return (Boston,)


@app.cell
def _(Boston):
    _X = pd.DataFrame({"intercept": np.ones(Boston.shape[0]), "lstat": Boston["lstat"]})
    _X[:4]
    return


@app.cell
def _(Boston, X):
    y = Boston["medv"]
    model = sm.OLS(y, X)
    results = model.fit()
    return results, y


@app.cell
def _(results):
    summarize(results=results)
    return


@app.cell
def _(Boston):
    design = MS(["lstat"])
    X = design.fit_transform(Boston)
    X[:4]
    return X, design


@app.cell
def _(results):
    results.summary()
    return


@app.cell
def _(results):
    results.params
    return


@app.cell
def _(design):
    new_df = pd.DataFrame({"lstat": [5, 10, 15]})
    newX = design.transform(new_df)
    newX
    return (newX,)


@app.cell
def _(newX, results):
    new_predictions = results.get_prediction(newX)
    new_predictions.predicted_mean
    return (new_predictions,)


@app.cell
def _(new_predictions):
    new_predictions.conf_int(alpha=0.05)
    return


@app.cell
def _(new_predictions):
    # observation, so includes random error too
    new_predictions.conf_int(obs=True, alpha=0.05)
    return


@app.function
def abline(ax, b, m, *args, **kwargs):
    "add a line with slope m and intercept b to axes obj"
    xlim = ax.get_xlim()
    ylim = [m * xlim[0] + b, m * xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)


@app.cell
def _(Boston, results):
    ax = Boston.plot.scatter("lstat", "medv")
    abline(ax, results.params[0], results.params[1], "r--", linewidth=3)
    ax
    return


@app.cell
def _(results):
    def _():
        ax = subplots(figsize=(8, 8))[1]
        ax.scatter(results.fittedvalues, results.resid)
        ax.set_xlabel("Fitted value ")
        ax.set_ylabel("Residual ")
        return ax.axhline(0, c="k", ls="--")

    _()
    return


@app.cell
def _(X, results):
    def _():
        infl = results.get_influence()
        ax = subplots(figsize=(8, 8))[1]
        ax.scatter(np.arange(X.shape[0]), infl.hat_matrix_diag)
        ax.set_xlabel("Index ")
        ax.set_ylabel("Leverage ")
        return np.argmax(infl.hat_matrix_diag), ax

    outa, outb = _()
    print("Leverage stat of worst point: ", outa)
    outb
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Multiple Linear Regression
    """)
    return


@app.cell
def _(Boston, y):
    mlr_X = MS(["lstat", "age"]).fit_transform(Boston)
    model1 = sm.OLS(y, mlr_X)
    results1 = model1.fit()
    summarize(results1)
    return (results1,)


@app.cell
def _(Boston):
    terms = Boston.columns.drop("medv")
    terms
    return (terms,)


@app.cell
def _(Boston, terms, y):
    mlr2_X = MS(terms).fit_transform(Boston)
    model2 = sm.OLS(y, mlr2_X)
    results2 = model2.fit()
    summarize(results2)
    return (mlr2_X,)


@app.cell
def _(mlr2_X):
    vals = [VIF(mlr2_X, i) for i in range(1, mlr2_X.shape[1])]
    vif = pd.DataFrame({"vif": vals}, index=mlr2_X.columns[1:])
    vif
    return


@app.cell
def _(Boston, y):
    nl_X = MS([poly("lstat", degree=2), "age"]).fit_transform(Boston)
    model3 = sm.OLS(y, nl_X)
    results3 = model3.fit()
    summarize(results3)
    return (results3,)


@app.cell
def _(results1, results3):
    # how much better with non-linear term for lstat included vs simple multiple linear regression
    anova_lm(results1, results3)
    return


@app.cell
def _(results3):
    nl_ax = subplots(figsize=(8, 8))[1]
    nl_ax.scatter(results3.fittedvalues, results3.resid)
    nl_ax.set_xlabel("Fitted Value")
    nl_ax.set_ylabel("Residual")
    nl_ax.axhline(0, c="k", ls="--")
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exercises
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

    Describe the null hypotheses to which the p-values given in Table 3.4 correspond. Explain what conclusions you can draw based on these p-values. Your explanation should be phrased in terms of sales, TV, radio, and newspaper, rather than in terms of the coefficients of the linear model.

    Answer:

    The null hypothesis is that the intercept is 0. Same for TV advertising spend, radio, and newspaper. It is the idea that it has no real effect and any fitted coefficient here is not statistically significant.

    We find that we can reject the null for intercept at 95% confidence, therefore the intercept is confidently not 0. For TV and radio spends, their impact on sales can confidently be stated as significant and non-zero. Not true for newspaper, where we cannot reject the null. It may well be zero effect.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 2

    Carefully explain differences between KNN classifier and KNN regression methods.

    Answer:

    KNN classifier is a clustering technique, which falls into unsupervised learning. We do not have predictions of a dependant variable which we would like to fit a model to. Instead, we only have a collection of independant variables, which we would like to group 'similar' observations by. As K increases, because more observations are included in each point of the fitted model, the fit is more linear and less flexible. Commonly used and useful as highly flexible clustering technique.

    KNN regression is similar in that the K nearest neighbouring observations are used (in cartesian space), to find the points of the fitted model. However it is a supervised regression technique. We do have a dependant variable, and the average of K nearest neighbours are being used to find predicted values for new values of independant variables. Uncommonly used as for problems with more than a few independant variables (high p) then linear regression tends to have less error while also provided better predictions.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3

    Suppose we have a data set with five predictors, X1 = GPA, X2 =
    IQ, X3 = Level (1 for College and 0 for High School), X4 = Interac-
    tion between GPA and IQ, and X5 = Interaction between GPA and
    Level. The response is starting salary after graduation (in thousands
    of dollars). Suppose we use least squares to fit the model, and get
    ˆβ0 = 50, ˆβ1 = 20, ˆβ2 = 0.07, ˆβ3 = 35, ˆβ4 = 0.01, ˆβ5 = −10.
    (a) Which answer is correct, and why?
    i. For a fixed value of IQ and GPA, high school graduates earn
    more, on average, than college graduates.
    ii. For a fixed value of IQ and GPA, college graduates earn
    more, on average, than high school graduates.
    128 3. Linear Regression
    iii. For a fixed value of IQ and GPA, high school graduates earn
    more, on average, than college graduates provided that the
    GPA is high enough.
    iv. For a fixed value of IQ and GPA, college graduates earn
    more, on average, than high school graduates provided that
    the GPA is high enough.
    (b) Predict the salary of a college graduate with IQ of 110 and a
    GPA of 4.0.
    (c) True or false: Since the coefficient for the GPA/IQ interaction
    term is very small, there is very little evidence of an interaction
    effect. Justify your answer.

    Answer:

    a) iii. because of the interaction term, if you make GPA high enough, you flip over to high school graduates earning more than college.

    b) 137,100 $

    c) True. Variance is likely much much higher for the salary than this coefficient's value, so it would have a high p-value and it would be very unlikely we could reject the null that the interation term is big. Very likely it is 0 and there is no effect, though there may not be enough evidence for that either.
    """)
    return


@app.cell
def _():
    print(50 + (4 * 20) + (110 * 0.07) + 35 + (0.01 * 4 * 110) + (-10 * 4))
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

    This question involves the use of simple linear regression on the Auto
    data set.
    (a) Use the sm.OLS() function to perform a simple linear regression
    with mpg as the response and horsepower as the predictor. Use
    the summarize() function to print the results. Comment on the
    output. For example:
    i. Is there a relationship between the predictor and the re-
    sponse?
    ii. How strong is the relationship between the predictor and
    the response?
    iii. Is the relationship between the predictor and the response
    positive or negative?
    iv. What is the predicted mpg associated with a horsepower of
    98? What are the associated 95 % confidence and prediction
    intervals?
    (b) Plot the response and the predictor in a new set of axes ax. Use
    the ax.axline() method or the abline() function defined in the
    lab to display the least squares regression line.
    (c) Produce some of diagnostic plots of the least squares regression
    fit as described in the lab. Comment on any problems you see
    with the fit.
    """)
    return


@app.cell
def _():
    # a
    Auto = load_data("Auto")
    Auto
    return (Auto,)


@app.cell
def _(Auto):
    design8 = MS(["horsepower"])
    X8 = design8.fit_transform(Auto)
    y8 = Auto["mpg"]
    results8 = sm.OLS(y8, X8).fit()
    summarize(results8)
    return (results8,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    1. there is a relationship, as p-value for the slope is 0
    2. very strong relationship
    3. it is a negative relationship, as horsepower increases, the mpg decreases
    4.
    """)
    return


@app.cell
def _(results8):
    x_for_pred = pd.DataFrame({0: [1], 1: [98]})
    pred_mpg = results8.get_prediction(x_for_pred)
    pred_mpg.predicted_mean
    return (pred_mpg,)


@app.cell
def _(pred_mpg):
    print(pred_mpg.conf_int(obs=False), pred_mpg.conf_int(obs=True))
    return


@app.cell
def _(Auto, results8):
    ax8 = Auto.plot.scatter("horsepower", "mpg", figsize=(8, 8))
    abline(ax8, results8.params[0], results8.params[1], "r--", linewidth=3)
    ax8.set_title("MPG vs horsepower")
    ax8.set_xlabel("Horsepower")
    ax8.set_ylabel("MPG")
    ax8
    return


@app.cell
def _(results8):
    infl8 = results8.get_influence()
    ax_resid = subplots(figsize=(8, 8))[1]
    ax_resid.scatter(results8.fittedvalues, results8.resid)
    ax_resid.axhline(0, c="k", ls="--")
    ax_resid.set_xlabel("Fitted values")
    ax_resid.set_ylabel("Residuals")
    ax_resid.set_title("Residual plot for simple regression")
    ax_resid
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The fitted line slopes downward strongly, which confirms the negative association between horsepower and mpg. The residual plot shows a curve rather than a random cloud, suggesting the relationship is not perfectly linear and that some non-linearity remains. A few high-leverage observations at the extremes are also visible.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 9

    This question involves the use of multiple linear regression on the Auto data set.

    Use mpg as the response and all of the other variables except name as predictors. Comment on the relationship between the predictors and the response, and compare the results with the simple linear regression from Question 8.
    """)
    return


@app.cell
def _():
    Auto = load_data("Auto")
    X9 = MS(
        [
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "year",
            "origin",
        ]
    ).fit_transform(Auto)
    y9 = Auto["mpg"]
    results9 = sm.OLS(y9, X9).fit()
    summarize(results9)
    return (results9,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The multiple regression fit usually shows a much stronger relationship than the simple horsepower-only model. In particular, predictors such as weight and year are often highly significant, and the model explains a large fraction of the variation in mpg. Compared with the simple regression, the multiple regression is more informative because it accounts for several variables at once rather than one predictor in isolation.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 10

    This question involves the use of regression on the Carseats data set.

    Fit a model predicting Sales from Price, Urban, and US, and then consider whether an interaction between Price and US is needed. Interpret the coefficient estimates in context.
    """)
    return


@app.cell
def _():
    Carseats = load_data("Carseats")
    Carseats["Urban"] = Carseats["Urban"].astype(str)
    Carseats["US"] = Carseats["US"].astype(str)
    X10 = MS(["Price", "Urban", "US"]).fit_transform(Carseats)
    y10 = Carseats["Sales"]
    results10 = sm.OLS(y10, X10).fit()
    summarize(results10)
    return (results10,)


@app.cell
def _(Carseats, results10):
    # interaction term between price and US
    X10_int = MS(["Price", "Urban", "US", "Price:US"]).fit_transform(Carseats)
    results10_int = sm.OLS(Carseats["Sales"], X10_int).fit()
    summarize(results10_int)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    An interaction term between Price and US is sensible because the effect of price may differ in the US versus outside the US. If the interaction is significant, the slope of Sales against Price is not the same in the two markets. In practice, this often gives a better fit than a purely additive model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 11

    This question involves the use of multiple linear regression and polynomial terms on the Boston data set.

    Fit a model for median house value using a nonlinear term in lstat together with another relevant predictor such as age or dis. Comment on the improvement over a linear model.
    """)
    return


@app.cell
def _():
    Boston = load_data("Boston")
    X11 = MS([poly("lstat", degree=2), "age"]).fit_transform(Boston)
    y11 = Boston["medv"]
    results11 = sm.OLS(y11, X11).fit()
    summarize(results11)
    return (results11,)


@app.cell
def _(results11, Boston):
    # Compare against the linear model using lstat and age only
    X11_lin = MS(["lstat", "age"]).fit_transform(Boston)
    results11_lin = sm.OLS(Boston["medv"], X11_lin).fit()
    anova_lm(results11_lin, results11)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The quadratic term in lstat is typically useful because the relationship between lower-status proportion and median home value is often curved rather than strictly linear. A significant improvement in fit is evidence that the nonlinear term captures structure that a simple linear model misses.
    """)
    return


if __name__ == "__main__":
    app.run()
