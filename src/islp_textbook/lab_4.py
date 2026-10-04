import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    import numpy as np
    import pandas as pd
    from matplotlib.pyplot import subplots
    import statsmodels.api as sm
    from ISLP import load_data
    from ISLP.models import ModelSpec as MS, summarize

    from ISLP import confusion_table
    from ISLP.models import contrast
    from sklearn.discriminant_analysis import (
        QuadraticDiscriminantAnalysis as QDA,
        LinearDiscriminantAnalysis as LDA,
    )
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    import plotly.express as px


@app.cell
def _():
    Smarket = load_data('Smarket')
    Smarket
    return (Smarket,)


@app.cell
def _(Smarket):
    Smarket.columns
    Smarket.dtypes
    return


@app.cell
def _(Smarket):
    Smarket_numerical_only = Smarket.drop(columns=['Direction'])
    corr_matrix = Smarket_numerical_only.corr()
    # heatmap plot of this correlation matrix
    px.heat(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title='Correlation Matrix Heatmap')
    return


@app.cell
def _(Smarket):
    Smarket.plot(y='Volume')
    return


@app.cell
def _(Smarket):
    allvars = Smarket.columns.drop(['Today', 'Direction', 'Year'])
    design = MS(allvars)
    X = design.fit_transform(Smarket)
    y = Smarket['Direction'] == 'Up'
    glm = sm.GLM(y, X, family=sm.families.Binomial())
    results = glm.fit()
    summarize(results)
    return X, results, y


@app.cell
def _(results):
    results.params
    return


@app.cell
def _(results):
    results.pvalues
    return


@app.cell
def _(results):
    probs = results.predict()
    probs[:10]
    return (probs,)


@app.cell
def _(probs):
    labels = np.array(['Down']*1250)
    labels[probs>0.5] = 'Up'
    labels[:10]
    return (labels,)


@app.cell
def _(Smarket, labels):
    confusion_table(labels, Smarket['Direction'])
    return


@app.cell
def _(Smarket, labels):
    np.mean(labels == Smarket['Direction'])
    return


@app.cell
def _(Smarket):
    train = (Smarket.Year < 2005)
    return (train,)


@app.cell
def _(Smarket, train):
    Smarket_train = Smarket.loc[train]
    Smarket_test = Smarket.loc[~train]
    Smarket_test.shape
    return


@app.cell
def _(Smarket, X, train, y):
    X_train, X_test = X.loc[train], X.loc[~train]
    y_train, y_test = y.loc[train], y.loc[~train]

    glm_train = sm.GLM(y_train, X_train, family=sm.families.Binomial())

    results2 = glm_train.fit()
    probs2 = results2.predict(exog=X_test)

    D = Smarket.Direction
    L_train, L_test = D.loc[train], D.loc[~train]

    labels2 = np.array(['Down']*252)
    labels2[probs2>0.5] = 'Up'

    confusion_table(labels2, L_test)
    return L_test, L_train, labels2, y_train


@app.cell
def _(L_test, labels2):
    np.mean(labels2 != L_test)
    return


@app.cell
def _(Smarket):
    model_minimal = MS(['Lag1', 'Lag2']).fit(Smarket)
    X_minimal = model_minimal.transform(Smarket)
    return X_minimal, model_minimal


@app.cell
def _(L_test, X_minimal, train, y_train):
    Xmin_train, Xmin_test = X_minimal.loc[train], X_minimal.loc[~train]
    glm_min_train = sm.GLM(y_train, Xmin_train, family=sm.families.Binomial())
    results_min = glm_min_train.fit()
    probs_min = results_min.predict(exog=Xmin_test)
    labels_min = np.array(['Down']*252)
    labels_min[probs_min>0.5] = 'Up'
    confusion_table(labels_min, L_test)
    return Xmin_test, Xmin_train, labels_min, results_min


@app.cell
def _(L_test, labels_min):
    np.mean(labels_min != L_test)
    return


@app.cell
def _(model_minimal, results_min):
    newdata = pd.DataFrame(
        {
            'Lag1': [1.2, 1.5],
            'Lag2': [1.1, -0.8]
        }
    )
    newX = model_minimal.transform(newdata)
    results_min.predict(newX)
    return


@app.cell
def _(L_train, Xmin_test, Xmin_train):
    lda = LDA(store_covariance=True)
    X_lda_train, X_lda_test = [
        matrix.drop(columns=['intercept'])
        for matrix in [Xmin_train, Xmin_test]
    ]
    lda.fit(X_lda_train, L_train)
    return X_lda_test, X_lda_train, lda


@app.cell
def _(lda):
    lda.means_
    return


@app.cell
def _(lda):
    lda.priors_
    return


@app.cell
def _(lda):
    lda.classes_
    return


@app.cell
def _(lda):
    lda.scalings_
    return


@app.cell
def _(L_test, X_lda_test, lda):
    lda_pred = lda.predict(X_lda_test)
    conf_table_lda = confusion_table(lda_pred, L_test)
    conf_table_lda
    return (lda_pred,)


@app.cell
def _(L_test, lda_pred):
    np.mean(lda_pred != L_test)
    return


@app.cell
def _(X_lda_test, lda, lda_pred):
    lda_prob = lda.predict_proba(X_lda_test)
    np.all(
        np.where(
            lda_prob[:,1] >= 0.5, 'Up', 'Down'
        ) == lda_pred
    )
    return


@app.cell
def _(L_train, X_lda_train):
    NB = GaussianNB()
    NB.fit(X_lda_train, L_train)
    NB.classes_
    return (NB,)


@app.cell
def _(NB):
    NB.class_prior_
    return


@app.cell
def _(NB):
    NB.theta_
    return


@app.cell
def _(NB):
    NB.var_
    return


@app.cell
def _(L_test, NB, X_lda_test):
    nb_labels = NB.predict(X_lda_test)
    conf_table_nb = confusion_table(nb_labels, L_test)
    conf_table_nb
    return (nb_labels,)


@app.cell
def _(L_test, nb_labels):
    np.mean(nb_labels != L_test)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
