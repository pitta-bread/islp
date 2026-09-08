from typing import Any

import numpy as np
import plotly.express as px
from dash import Dash, dcc, html
from matplotlib.pyplot import subplots
from numpy.random._generator import Generator
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots
from plotly.tools import mpl_to_plotly


def main() -> None:
    chap2_3_3()
    figures: list[Figure] = chap2_3_4()

    app = Dash(__name__)
    app.layout = html.Div([dcc.Graph(figure=figure) for figure in figures])
    app.run(debug=True)


def chap2_3_3() -> None:
    print("--- Chapter 2.3.3 ---")
    # default rng
    rng: Generator = np.random.default_rng(1303)

    x: np.ndarray = np.array([3, 4, 5])
    y: np.ndarray = np.array([4, 9, 7])

    z: np.ndarray = x + y
    print("z: ", z)

    x: np.ndarray = np.array([[1, 2], [3, 4]])
    print("x:\n", x)
    print("x dtype: ", x.dtype)
    print("x shape: ", x.shape)

    x: np.ndarray = np.array([1, 2, 3, 4, 5, 6])
    print("beginning x:\n", x)
    x_reshape: np.ndarray = x.reshape((2, 3))
    print("reshaped x:\n", x_reshape)
    x_reshape[0, 0] = 5
    print("x_reshape after we modify its top left element:\n", x_reshape)

    print("x_reshape shape: ", x_reshape.shape)
    print("x_reshape dtype: ", x_reshape.dtype)
    print("x_reshape ndim: ", x_reshape.ndim)
    print("x_reshape transpose: ", x_reshape.T)

    print("random x array of size 10:")
    x: np.ndarray = rng.normal(size=10)
    print(x)
    y: np.ndarray = x + rng.normal(loc=50, scale=1, size=10)
    print("random y array of size 10:\n", y)
    print("correlation coefficient of x and y:\n", np.corrcoef(x, y))

    print("mean of y: ", np.mean(y))

    X: np.ndarray = rng.standard_normal((10, 3))
    print("X:\n", X)
    print("mean of X along axis 0: ", np.mean(X, axis=0))


def chap2_3_4() -> list[Figure]:
    print("--- Chapter 2.3.4 ---")

    rng: Generator = np.random.default_rng(seed=1303)
    x: np.ndarray = rng.standard_normal(size=100)
    y: np.ndarray = rng.standard_normal(size=100)

    fig1: Figure = px.scatter(
        x=x,
        y=y,
        title="Plot of X vs Y",
        labels={"x": "X-axis", "y": "Y-axis"},
    )

    fig2: Figure = px.scatter(
        x=x,
        y=y,
        title="Plot of X vs Y with Regression Line",
        labels={"x": "X-axis", "y": "Y-axis"},
        trendline="ols",
    )

    fig3: Figure = px.scatter(
        x=x,
        y=y,
        title="Plot of X vs Y with Regression Line RED",
        labels={"x": "X-axis", "y": "Y-axis"},
        trendline="ols",
        trendline_color_override="red",
    )

    side_by_side: Figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(fig1.layout.title.text, fig2.layout.title.text),
    )

    for trace in fig1.data:
        side_by_side.add_trace(trace, row=1, col=1)
    for trace in fig2.data:
        side_by_side.add_trace(trace, row=1, col=2)

    side_by_side.update_xaxes(title_text="X-axis", row=1, col=1)
    side_by_side.update_xaxes(title_text="X-axis", row=1, col=2)
    side_by_side.update_yaxes(title_text="Y-axis", row=1, col=1)
    side_by_side.update_yaxes(title_text="Y-axis", row=1, col=2)

    fig, ax = subplots(figsize=(8, 8))
    x: np.ndarray = np.linspace(start=-np.pi, stop=np.pi, num=50)
    y: np.ndarray = x
    f: np.ndarray[Any] = np.multiply.outer(np.cos(y), 1 / (1 + x**2))
    ax.contour(x, y, f, levels=45)
    plotly_contour_fig: Figure = mpl_to_plotly(fig)

    return [side_by_side, fig3, plotly_contour_fig]
