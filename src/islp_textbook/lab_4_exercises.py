import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["ipynb"])

with app.setup:
    import numpy as np
    import statsmodels.api as sm
    from ISLP import confusion_table, load_data
    from ISLP.models import ModelSpec as MS
    from ISLP.models import summarize
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis as LDA,
    )
    from sklearn.discriminant_analysis import (
        QuadraticDiscriminantAnalysis as QDA,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler


@app.cell
def _():
    Weekly = load_data("Weekly")
    Weekly
    return (Weekly,)


@app.cell
def _(Weekly):
    Weekly.describe()
    return


@app.cell
def _(Weekly):
    Weekly.Volume.plot()
    return


@app.cell
def _(Weekly):
    X = MS(Weekly.columns.drop(["Today", "Year", "Direction"])).fit_transform(Weekly)
    y = Weekly.Direction == "Up"

    model = sm.GLM(y, X, family=sm.families.Binomial())
    results = model.fit()
    summarize(results)
    return X, results, y


@app.cell
def _(y):
    y
    return


@app.cell
def _(X, results):
    y_hat = results.predict(X)
    y_hat
    return (y_hat,)


@app.cell
def _(y_hat):
    y_hat_labels = y_hat > 0.5
    y_hat_labels
    return (y_hat_labels,)


@app.cell
def _(y, y_hat_labels):
    print("too many false positives")
    confusion_table(y_hat_labels, y)
    return


@app.cell
def _(y, y_hat_labels):
    np.mean(y_hat_labels == y)
    return


@app.cell
def _():
    fresh_data = load_data("Weekly")

    def slice_train_then_test(sklearn_model, fresh_data, random_state=42):
        dataset = fresh_data.copy()
        train_slice = dataset.Year < 2009
        test_slice = dataset.Year >= 2009
        X_train = dataset[train_slice].drop(
            columns=[
                "Today",
                "Year",
                "Direction",
                "Lag1",
                "Lag3",
                "Lag4",
                "Lag5",
                "Volume",
            ]
        )
        X_test = dataset[test_slice].drop(
            columns=[
                "Today",
                "Year",
                "Direction",
                "Lag1",
                "Lag3",
                "Lag4",
                "Lag5",
                "Volume",
            ]
        )
        y_train = dataset[train_slice].Direction == "Up"
        y_test = dataset[test_slice].Direction == "Up"

        model = sklearn_model.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        y_hat_labels = y_hat > 0.5
        return confusion_table(y_hat_labels, y_test), np.mean(y_hat == y_test)

    models = [
        LogisticRegression(),
        LDA(),
        QDA(),
        KNeighborsClassifier(n_neighbors=1),
        GaussianNB(),
    ]

    print(
        f"Always-Up baseline: {(fresh_data[fresh_data.Year >= 2009].Direction == 'Up').mean():.4f}\n"
    )
    for m in models:
        print(f"Evaluating model: {m.__class__.__name__}")
        conf_table, accuracy = slice_train_then_test(m, fresh_data)
        print(conf_table)
        print(f"Accuracy: {accuracy:.4f}\n")
    return (fresh_data,)


@app.cell
def _(fresh_data):
    # Now let's try every combination possible of predictors, first we'll need to adapt our slice_train_then_test function to not do dropping and assume the dataset is already in the right format.

    def slice_train_then_test_2(sklearn_model, dataset):
        train_slice = dataset.Year < 2009
        test_slice = dataset.Year >= 2009
        X_train = dataset[train_slice].drop(columns=["Year", "Direction"])
        X_test = dataset[test_slice].drop(columns=["Year", "Direction"])
        y_train = dataset[train_slice].Direction == "Up"
        y_test = dataset[test_slice].Direction == "Up"

        model_fitted = sklearn_model.fit(X_train, y_train)
        y_hat = model_fitted.predict(X_test)
        y_hat_labels = y_hat > 0.5
        return confusion_table(y_hat_labels, y_test), np.mean(y_hat == y_test)

    # Now a function which will generate all combinations of predictors and evaluate them with the given model
    from itertools import combinations

    def evaluate_all_combinations(sklearn_model, dataset):
        predictors = dataset.columns.drop(["Today", "Year", "Direction"])
        results = []

        for r in range(1, len(predictors) + 1):
            for combo in combinations(predictors, r):
                subset = dataset[list(combo) + ["Year", "Direction"]]
                conf_table, accuracy = slice_train_then_test_2(sklearn_model, subset)
                results.append(
                    {
                        "predictors": combo,
                        "confusion_table": conf_table,
                        "accuracy": accuracy,
                    }
                )

        return results

    enhanced_data = fresh_data.copy()
    # Now we include an interaction term for Lag1 and Lag2
    enhanced_data["Lag1_Lag2"] = enhanced_data["Lag1"] * enhanced_data["Lag2"]
    # Volume is strictly positive, so no offset is needed
    enhanced_data["LogVolume"] = np.log(enhanced_data["Volume"])

    new_models = [
        LogisticRegression(),
        LDA(),
        QDA(),
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3)),
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10)),
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=20)),
        GaussianNB(),
    ]

    print(
        f"Always-Up baseline: {(enhanced_data[enhanced_data.Year >= 2009].Direction == 'Up').mean():.4f}\n"
    )
    overall = []
    for m2 in new_models:
        print(f"Evaluating all combinations for model: {m2}")
        results2 = evaluate_all_combinations(m2, enhanced_data)
        best_result = max(results2, key=lambda x: x["accuracy"])
        overall.append((best_result["accuracy"], str(m2), best_result))
        print(f"Best combination of predictors: {best_result['predictors']}")
        print(f"Confusion Table:\n{best_result['confusion_table']}")
        print(f"Accuracy: {best_result['accuracy']:.4f}\n")

    top_acc, top_name, top = max(overall, key=lambda t: t[0])
    print(f"Best overall: {top_name} on {top['predictors']}, accuracy {top_acc:.4f}")
    print(top["confusion_table"])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
