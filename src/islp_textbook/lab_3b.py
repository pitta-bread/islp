import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from ISLP import load_data
    import plotly.express as px
    from plotly.subplots import make_subplots
    from statsmodels.stats.outliers_influence import (
        variance_inflation_factor as VIF,
    )
    from statsmodels.stats.anova import anova_lm
    from ISLP.models import summarize, poly
    from ISLP.models import ModelSpec as MS


@app.cell
def _():
    Boston = load_data('Boston')
    Boston
    return (Boston,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Boston Housing Data Dictionary

    | Variable | Description |
    | :--- | :--- |
    | **`crim`** | Per capita crime rate by town |
    | **`zn`** | Proportion of residential land zoned for lots over 25,000 sq. ft. |
    | **`indus`** | Proportion of non-retail business acres per town |
    | **`chas`** | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) |
    | **`nox`** | Nitrogen oxides concentration (parts per 10 million) |
    | **`rm`** | Average number of rooms per dwelling |
    | **`age`** | Proportion of owner-occupied units built prior to 1940 |
    | **`dis`** | Weighted mean of distances to five Boston employment centres |
    | **`rad`** | Index of accessibility to radial highways |
    | **`tax`** | Full-value property-tax rate per \$10,000 |
    | **`ptratio`** | Pupil-teacher ratio by town |
    | **`lstat`** | Lower status of the population (percent) |
    | **`medv`** | Median value of owner-occupied homes in \$1,000s |
    """)
    return


@app.cell
def _(Boston):
    target = Boston.columns[0]  # 'crim'
    predictors = Boston.columns[1:]

    simple_models = {}
    _summary_rows = []

    for _col in predictors:
        _X = MS([_col]).fit_transform(Boston)
        _y = Boston[target]
        _fit = sm.OLS(_y, _X).fit()
        simple_models[_col] = _fit

        _s = summarize(_fit)
        _summary_rows.append(
            {
                "predictor": _col,
                "intercept": _s.loc["intercept", "coef"],
                "coef": _s.loc[_col, "coef"],
                "std_err": _s.loc[_col, "std err"],
                "t_stat": _s.loc[_col, "t"],
                "p_value": _s.loc[_col, "P>|t|"],
            }
        )

    regression_summaries = pd.DataFrame(_summary_rows)
    regression_summaries
    return predictors, regression_summaries, simple_models, target


@app.cell
def _(regression_summaries):
    px.bar(regression_summaries, x='predictor', y='p_value', title='P-values of Predictors in Simple Linear Regression Models', labels={'p_value': 'P-value', 'predictor': 'Predictor'})
    return


@app.cell
def _(Boston):
    # taking indus (non-retail proportion of lots in town) as an example, visualise with plotly express
    crim_vs_indus_df = Boston[['crim', 'indus']].copy()
    px.scatter(crim_vs_indus_df, x='indus', y='crim', trendline='ols', title='Crime Rate vs Non-Retail Proportion of Lots in Town')
    return


@app.cell
def _(Boston):
    _predictors = Boston.columns[1:]
    grid_fig = make_subplots(
        rows=4,
        cols=3,
        subplot_titles=[f"crim vs {_col}" for _col in _predictors],
        vertical_spacing=0.07,
        horizontal_spacing=0.06,
    )

    for _i, _col in enumerate(_predictors):
        _row = (_i // 3) + 1
        _col_idx = (_i % 3) + 1
        _sub_fig = px.scatter(
            Boston,
            x=_col,
            y="crim",
            trendline="ols",
            trendline_color_override="red",
        )
        for _trace in _sub_fig.data:
            _trace.showlegend = False
            grid_fig.add_trace(_trace, row=_row, col=_col_idx)

        grid_fig.update_xaxes(title_text=_col, row=_row, col=_col_idx)
        grid_fig.update_yaxes(title_text="crim", row=_row, col=_col_idx)

    grid_fig.update_layout(
        height=1100,
        width=950,
        title_text="Simple Linear Regressions: Crime Rate (crim) vs Each Predictor",
        title_x=0.5,
    )
    grid_fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    There is only one predictor, which when analysed in isolation, does not have a statistically significant relationship (at 95% conf level) with crime rate. This is 'chas' i.e. the dummy variable for a town containing the river or not (boolean caategory).
    """)
    return


@app.cell
def _(Boston, predictors, target):
    # all predictors with MLR
    _X = MS(predictors).fit_transform(Boston)
    _y = Boston[target]
    mlr_fit = sm.OLS(_y, _X).fit()
    mlr_summary = summarize(mlr_fit)
    mlr_summary
    return (mlr_summary,)


@app.cell
def _(mlr_summary):
    for idx, row in mlr_summary.iterrows():
        if idx != "intercept" and row["P>|t|"] < 0.05:
            print(
                f"We can reject the null that there is no relationship at 95% confidence level for predictor '{idx}'."
            )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This is a clear instance where due to colinearity or synergy between predictors, some predictors that were significant in isolation are no longer significant when considered together in a multiple linear regression model. This highlights the importance of considering the relationships between predictors when building regression models.
    """)
    return


@app.cell
def _(mlr_summary):
    mlr_summaries = mlr_summary.drop("intercept")[["coef"]]
    mlr_summaries
    return (mlr_summaries,)


@app.cell
def _(mlr_summaries, regression_summaries):
    merged_summaries = pd.merge(
        regression_summaries[["predictor", "coef"]],
        mlr_summaries,
        left_on="predictor",
        right_index=True,
        suffixes=("_simple", "_mlr"),
    )
    merged_summaries
    return (merged_summaries,)


@app.cell
def _(merged_summaries):
    px.scatter(
        merged_summaries,
        x="coef_simple",
        y="coef_mlr",
        text="predictor",
        title="Comparison of Coefficients: Simple vs Multiple Linear Regression Models",
        labels={"coef_simple": "Coefficient (Simple LR)", "coef_mlr": "Coefficient (MLR)"},
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Comparison: Simple vs. Multiple Regression Coefficients

    1. **The Primary Outlier (`nox`)**:
       - In simple linear regression, **`nox`** has a large positive coefficient ($\hat{\beta} \approx 31.25$, $p < 10^{-50}$). In isolation, higher nitrogen oxide concentration strongly correlates with higher crime rate, largely acting as a surrogate for dense, industrialized urban areas.
       - In multiple linear regression, when controlling for all other predictors (such as `dis`, `rad`, `indus`, and `age`), the coefficient swings dramatically to $\hat{\beta} \approx -9.96$ ($p \approx 0.060$). This swing of over **41 units** is by far the largest shift in the dataset, demonstrating how collinearity can dramatically alter univariate estimates.

    2. **Sign Reversals (`rm`, `ptratio`, `zn`)**:
       - **`rm`**: Shifts from $-2.68$ (simple) to $+0.63$ (MLR), likely due to collinearity with socio-economic status (`lstat`) and home value (`medv`).
       - **`ptratio`**: Shifts from $+1.15$ (simple) to $-0.30$ (MLR).
       - **`zn`**: Shifts from $-0.0739$ (simple) to $+0.0457$ (MLR).

    3. **Collinearity and Coefficient Shrinkage**:
       - Rather than coefficients being identical, most predictors see their coefficients shrink towards zero in the multiple regression model (e.g., `indus`, `age`, `tax`).
       - While 11 out of 12 predictors were statistically significant on their own, only **4 predictors (`zn`, `dis`, `rad`, `medv`)** remain significant at $\alpha = 0.05$ when all predictors are included simultaneously.
    """)
    return


@app.cell
def _(Boston, predictors, simple_models, target):
    _poly_rows = []
    polynomial_models = {}

    for _col in predictors:
        if _col == "chas":
            _poly_rows.append(
                {
                    "Predictor": _col,
                    "Linear (X) p-val": simple_models[_col].pvalues.iloc[-1],
                    "Quadratic (X^2) p-val": np.nan,
                    "Cubic (X^3) p-val": np.nan,
                    "ANOVA (vs Linear) p-val": np.nan,
                    "Evidence of Non-linearity": "No (binary categorical)",
                }
            )
            continue

        _X_poly = MS([poly(_col, degree=3)]).fit_transform(Boston)
        _fit_poly = sm.OLS(Boston[target], _X_poly).fit()
        polynomial_models[_col] = _fit_poly

        _X_lin = MS([poly(_col, degree=1)]).fit_transform(Boston)
        _fit_lin = sm.OLS(Boston[target], _X_lin).fit()
        _anova = anova_lm(_fit_lin, _fit_poly)
        _anova_p = _anova["Pr(>F)"].iloc[1]

        _p_lin = _fit_poly.pvalues.iloc[1]
        _p_quad = _fit_poly.pvalues.iloc[2]
        _p_cube = _fit_poly.pvalues.iloc[3]

        if _p_quad < 0.05 and _p_cube < 0.05:
            _verdict = "Yes (quadratic & cubic)"
        elif _p_quad < 0.05:
            _verdict = "Yes (quadratic)"
        elif _p_cube < 0.05:
            _verdict = "Yes (cubic)"
        elif _anova_p < 0.05:
            _verdict = "Yes (joint ANOVA)"
        else:
            _verdict = "No"

        _poly_rows.append(
            {
                "Predictor": _col,
                "Linear (X) p-val": _p_lin,
                "Quadratic (X^2) p-val": _p_quad,
                "Cubic (X^3) p-val": _p_cube,
                "ANOVA (vs Linear) p-val": _anova_p,
                "Evidence of Non-linearity": _verdict,
            }
        )

    nonlinear_summary_df = pd.DataFrame(_poly_rows)

    mo.vstack(
        [
            mo.md(r"""
            ## (d) Non-Linear Associations with Crime Rate ($Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \epsilon$)

            For each predictor, we fit a cubic polynomial model using orthogonal polynomials and compare it against the simple linear model using ANOVA:
            """),
            nonlinear_summary_df,
            mo.md(r"""
            ### Conclusion & Summary:
            - **`chas`**: As a qualitative binary indicator variable ($0$ or $1$), $X^2 = X^3 = X$, so higher-order polynomial terms cannot be fitted (rank deficient).
            - **All other 11 predictors (`zn`, `indus`, `nox`, `rm`, `age`, `dis`, `rad`, `tax`, `ptratio`, `lstat`, `medv`)**: There is **statistically significant evidence of a non-linear association** with crime rate ($p < 0.05$ on the ANOVA $F$-test and individual polynomial terms):
              - **Both quadratic ($X^2$) and cubic ($X^3$) terms significant**: `indus`, `nox`, `age`, `dis`, `ptratio`, and `medv`.
              - **Quadratic ($X^2$) term significant**: `zn`, `rm`, `rad`, `tax`, and `lstat`.
            """),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
