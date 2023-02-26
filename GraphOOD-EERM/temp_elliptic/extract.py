import pandas as pd
from collections import defaultdict
from argparse import Namespace
from numpy import array
import numpy as np
import os


df = defaultdict(list)
for ii, file in enumerate(os.listdir()):
    if 'out' not in file:
        continue
    args = None
    accs = []
    with open(file, 'r') as f:

        for line in f.readlines():
            if 'Namespace(c' in line:
                args = eval(line)
            marker = 'Final Test'
            if marker in line:
                if '±' in line: # filter the all runs stats
                    continue
                line = line.split(marker)[-1]
                id_graph, line = line.split(':')
                res = eval(line)
                df[f'T{id_graph}'].append(res)
                accs.append(res)

                if id_graph == 8:
                    accs = []

df = pd.DataFrame.from_dict(df, orient='index').transpose()

import ipdb
ipdb.set_trace()

df['Avg'] = df.iloc[:, -9:].mean(1)

def fun(x):
    x = f'{x:.2f}'
    return x

merged = []
# metrics = list(df.columns[3:])
metrics = ['Avg']
for m in metrics:
    acc = df.groupby(list(df.columns[:-10]))[m].apply(np.mean)
    std = df.groupby(list(df.columns[:-10]))[m].apply(np.std).rename('std')
    # merged += [acc, std]
    merged += [acc.apply(fun) + '±' + std.apply(fun)]
new_df = pd.concat(merged, axis=1)
# new_df.columns = columns
# def fun(x): return float(x[:5])
# new_df.applymap(fun)#.drop(, axis=1)
# new_df['Avg'] = new_df.applymap(fun).iloc[:, :-1].mean(1)

df2 = new_df.reset_index()
for m in df2.gnn.unique():
    print(df2[df2.gnn==m].sort_values(by=[0],ascending=False).values[0])

import ipdb
ipdb.set_trace()

print(new_df)

