import pickle
import numpy as np
import pandas
from plotly import figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express.colors as pxcol


data_set = 1
param_names = ["V_c", "K_cl"]
with open("Images/PK_sim/PK_comp1/large_data_" + str(data_set) + "_2pop/pop_V_c_ll_profiles_2nd_derivs_data.pkl", "rb") as fp:
    pickle_data = pickle.load(fp)
max_res = dict(zip(param_names, -np.inf*np.ones((len(param_names), 100)))) # -np.inf*np.ones(len(pickle_data[("V_c", 0)]['Result'][0]))
x = {}
last_param = None
for graph_id, data in pickle_data.items():
    param, method, extrap = graph_id
    res = data['Result']
    max_res[param] = np.fmax(res[1], max_res[param])
    x[param] = res[0]

fig = make_subplots(rows=4, cols=2, row_heights=[0.55, 0.15, 0.15, 0.15], shared_xaxes=True)

for param in param_names:
    fig.add_trace(go.Scatter(
        x=x[param],
        y=max_res[param],
        name="Maximum of results",
        line=dict(color="black", width=2.0)
    ), row=1, col=param_names.index(param)+1)

time_table_data = np.zeros((3, 9))
error_table_data = np.zeros((3, 9))
last_param = None
prev_method = None
method_arg = -1
method_names = []
colours = pxcol.qualitative.Safe
for param, method, extrap in pickle_data.keys():
    data = pickle_data[(param, method, extrap)]
    time = data['Time']
    time_min = time/60
    if method != prev_method:
        method_arg += 1
        prev_method = method
        line_colour = colours.pop(0)
    if method == "Nelder-Mead":
        method = "N-M"
    method_name = method + ", " + str(extrap)
    if not method_name in method_names:
        method_names.append(method_name)
    method_pos = extrap + method_arg*3
    time_table_data[param_names.index(param), method_pos] = time_min
    res = data['Result']
    error = res[1]-max_res[param]
    if extrap == 2:
        fig.add_trace(go.Scatter(
            x=x[param],
            y=res[1],
            name="Results",
            line=dict(width=1, color=line_colour),
            legendgroup=method,
            legendgrouptitle_text=method,
        ), row=1, col=param_names.index(param)+1)

        fig.add_trace(go.Scatter(
            x=x[param],
            y=error,
            name="Error",
            line=dict(width=1, color=line_colour),
            legendgroup=method,
            legendgrouptitle_text="method",
        ), row=method_arg+2, col=param_names.index(param)+1)
    error_table_data[param_names.index(param), method_pos] = np.sum(error)**2


# fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
#               row=1, col=2)

fig.update_layout(
    width=900,
    height=900
)
fig.write_image("Images/PK_sim/PK_comp1/large_data_" + str(data_set) + "_2pop/pop_V_c_ll_profiles_method_error_graph.svg")
fig.show()

param_names.append("Total")
time_table_data[2, :] = np.sum(time_table_data, axis=0)
error_table_data[2, :] = np.sum(error_table_data, axis=0)

table_df = pandas.DataFrame(time_table_data, index=param_names, columns=method_names)
table_df = table_df.round(4)

fig = ff.create_table(table_df, index=True)
fig.update_layout(
    width=900,
    height=90
)
fig.write_image("Images/PK_sim/PK_comp1/large_data_" + str(data_set) + "_2pop/pop_V_c_ll_profiles_2nd_derivs_time_table.svg")
fig.show()

table_df = pandas.DataFrame(error_table_data, index=param_names, columns=method_names)
table_df = table_df.round(4)

fig = ff.create_table(table_df, index=True)
fig.update_layout(
    width=900,
    height=90
)
fig.write_image("Images/PK_sim/PK_comp1/large_data_" + str(data_set) + "_2pop/pop_V_c_ll_profiles_2nd_derivs_error_table.svg")
fig.show()
