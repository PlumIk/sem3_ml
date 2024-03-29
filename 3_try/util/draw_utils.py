import plotly

from .util import CLASS_NUMBER
import plotly.graph_objs as go
import plotly.figure_factory as ff


def draw_cost(cost):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[i + 1 for i in range(len(cost))], y=cost))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title="cost",
                      xaxis_title="batch",
                      yaxis_title="cross entropy",
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.show()
    return fig


def draw_conf_matrix(confusion_matrix):
    x = [i for i in range(CLASS_NUMBER)]
    y = x
    z_text = [[str(y) for y in x] for x in confusion_matrix]
    fig = ff.create_annotated_heatmap(confusion_matrix, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                      )
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    fig.update_layout(margin=dict(t=50, l=200))
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 9

    fig['data'][0]['showscale'] = True
    fig.show()
    return fig


def draw_roc_curves(roc_data_list, cls_names):
    fig = go.Figure()
    for i in range(len(roc_data_list)):
        fpr, tpr, auc = roc_data_list[i]
        _draw_roc_curve(fpr, tpr, auc, cls_names[i], fig)
    fig.add_trace(go.Scatter(name="bisector", x=[i / 100 for i in range(101)], y=[i / 100 for i in range(101)],
                             mode="markers"))
    fig.update_traces(showlegend=True)
    fig.show()
    return fig


def _draw_roc_curve(fpr, tpr, auc, cls_name, fig):
    fig.add_trace(go.Scatter(name=cls_name + " (auc = " + str(auc) + ")", x=fpr, y=tpr))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title="roc curve",
                      xaxis_title="fpr",
                      yaxis_title="tpr",
                      margin=dict(l=0, r=0, t=30, b=0))


def draw_roc_curve(fpr, tpr, auc, cls_name):
    fig = go.Figure()
    _draw_roc_curve(fpr, tpr, auc, cls_name, fig)
    fig.update_traces(showlegend=True)
    fig.add_trace(go.Scatter(name="bisector", x=[i / 100 for i in range(101)], y=[i / 100 for i in range(101)],
                             mode="markers"))
    fig.show()
    return fig


def save_param_to_html(fig, parameters_file_name, param_name):
    plotly.io.write_html(fig=fig, file=f'{parameters_file_name}_{param_name}')


def draw_metric_bar(cls, value, metric_name):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cls, y=value))
    fig.update_layout(legend_orientation="h",
                      legend=dict(x=.5, xanchor="center"),
                      title=metric_name,
                      xaxis_title="classes",
                      yaxis_title="value of " + metric_name,
                      margin=dict(l=0, r=0, t=30, b=0))
    fig.show()
    return fig
