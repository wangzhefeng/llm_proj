# -*- coding: utf-8 -*-

# ***************************************************
# * File        : my_app.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2024-08-06
# * Version     : 0.1.080621
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


# ------------------------------
# magic command
# ------------------------------
df = pd.DataFrame({
    "first column": [1, 2, 3, 4],
    "second column": [10, 20, 30, 40]
})
df

# ------------------------------
# st.write
# ------------------------------
st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    "first column": [1, 2, 3, 4],
    "second column": [10, 20, 30, 40]
}))


# ------------------------------
# st.dataframe
# ------------------------------
dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)


# ------------------------------
# Pandas Styler
# ------------------------------
dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns = ('col %d' % i for i in range(20))
)
st.dataframe(dataframe.style.highlight_max(axis = 0))

# ------------------------------
# static table
# ------------------------------
dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20))
)
st.table(dataframe.style.highlight_max(axis = 0))


# ------------------------------
# line chart
# ------------------------------
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns = ["a", "b", "c"]
)
st.line_chart(chart_data)

# ------------------------------
# st.map
# ------------------------------
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns = ['lat', 'lon']
)
st.map(map_data)

# ------------------------------
# show progress
# ------------------------------
import time

"Starting a long computation..."

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    # Update the progress bar with each iteration
    latest_iteration.text(f"Iteration {i + 1}")
    bar.progress(i + 1)
    time.sleep(0.1)

"...and now we\'re done!"


# ------------------------------
# st.checkbox
# ------------------------------
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns = ['a', 'b', 'c'])

    chart_data

# ------------------------------
# st.selectbox
# ------------------------------
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

option = st.selectbox(
    'Which number do you like best?',
     df['first column']
)

'You selected: ', option


# ------------------------------
# sidebar
# st.sidebar.selectbox
# st.sidebar.slider
# ------------------------------
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    label = "How would you like to be contacted?",
    options = ("Email", "Home phone", "Mobile phone")
)

# Add a slider to the sidebar
add_slider = st.sidebar.slider(
    label = "Select a range of values",
    min_value = 0.0, max_value = 100.0, 
    value = (25.0, 75.0)
)


# ------------------------------
# st.columns, st.expander
# ------------------------------
left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button("Press me!")

# or even better, call Streamlit functions inside a "with" block
with right_column:
    chosen = st.radio(
        "Sorting hat",
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin")
    )
    st.write(f"You are in {chosen} house!")






# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
