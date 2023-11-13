import pickle
import numpy as np
import pandas
from plotly import figure_factory as ff

data_set = 1
with open("Images/PK_sim/PK_comp1/large_data_" + str(data_set) + "_2pop/pop_V_c_ll_profiles_methods_trans_ind_data.pkl", "rb") as fp:
    pickle_data = pickle.load(fp)
max_res = -np.inf*np.ones(len(pickle_data['Nelder-Mead']['Result'][0]))
for data in pickle_data.values():
    res = data['Result']
    max_res = np.fmax(res[1], max_res)

tot_time = [None] * 3
time_table_data = np.zeros((3, 5))
for method in pickle_data.keys():
    data = pickle_data[method]
    time = data.pop('Time')
    print(method, time)
    if isinstance(time, str):
        time_str = time.split()
        i = -1
        time_min = 0
        print(time_str)
        for time_value in time_str[-2::-2]:
            time_min += float(time_value)*(60**i)
            i+=1
    else:
        time_min = time/60
    print(time_min)
    data['Time, min'] = time_min
    res = data.pop('Result')
    data["Error"] = np.sum((res[1]-max_res)**2)


table_df = pandas.DataFrame(pickle_data)
table_df = table_df.round(4)

fig = ff.create_table(table_df, index=True)
fig.update_layout(
    width=1000,
    height=65
)
fig.write_image("Images/PK_sim/PK_comp1/large_data_" + str(data_set) + "_2pop/pop_V_c_ll_profiles_methods_table.svg")
fig.show()
