import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import numpy as np
import io
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.font_manager as fm
import re
import plotly.express as px
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import shap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号

def evaluate_model(y_true, y_pred):
    """
    统一模型评估函数
    返回包含RMSE、MAE、R²、MSE、准确率的字典
    """
    # 计算回归指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 计算分类准确率（仅当目标变量为分类时）
    try:
        acc = accuracy_score(y_true, np.round(y_pred))
    except ValueError:
        acc = None  # 如果目标变量是连续值，则返回None
        
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MSE': mse,
        'Accuracy': acc
    }

# 定义 TimeSeriesDataset 类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length

    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.sequence_length],
            self.y[idx + self.sequence_length - 1]
        )

# 添加 LSTMModel 类定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

# 添加 GRUModel 类定义
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        predictions = self.fc(gru_out[:, -1, :])
        return predictions
    
def shap_analysis(model, X_train, X_test, feature_columns):
    """
    使用 SHAP 值分析模型的预测结果
    """
    # 根据模型类型选择解释器
    if isinstance(model, (xgb.XGBModel, RandomForestRegressor, lgb.LGBMModel)):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    elif isinstance(model, torch.nn.Module):
        explainer = shap.DeepExplainer(model, torch.tensor(X_train, dtype=torch.float32))
        shap_values = explainer.shap_values(torch.tensor(X_test, dtype=torch.float32))
    elif isinstance(model, SVR):
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
    else:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)


    # 可视化全局特征重要性图
    st.subheader("全局特征重要性")
    shap_values_summary = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': shap_values_summary
    }).sort_values(by='importance', ascending=False)

    fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title='全局特征重要性')
    st.plotly_chart(fig)
def display_metrics(train_metrics, test_metrics):
    """
    显示训练集和测试集的评估指标
    """
    # 格式化准确率显示
    train_acc = train_metrics['Accuracy'] if train_metrics['Accuracy'] is not None else 'N/A'
    test_acc = test_metrics['Accuracy'] if test_metrics['Accuracy'] is not None else 'N/A'

    # 显示评估指标
    st.write(f"""
    ## 模型评估指标
    ### 训练集
    - RMSE: {train_metrics['RMSE']:.4f}
    - MAE: {train_metrics['MAE']:.4f}
    - R²: {train_metrics['R²']:.4f}
    - MSE: {train_metrics['MSE']:.4f}
    - 准确率: {train_acc if isinstance(train_acc, str) else f"{train_acc:.2%}"}

    ### 测试集  
    - RMSE: {test_metrics['RMSE']:.4f}
    - MAE: {test_metrics['MAE']:.4f}
    - R²: {test_metrics['R²']:.4f}
    - MSE: {test_metrics['MSE']:.4f}
    - 准确率: {test_acc if isinstance(test_acc, str) else f"{test_acc:.2%}"}
    """)

# 设置页面布局
st.set_page_config(layout="wide", page_title="Machine Learning", page_icon="📈")
# 设置应用标题
st.title("数据预处理与模型训练 Web 应用 V2.7")

# 创建一个输入框来获取header的值
header = st.sidebar.text_input("请输入数据表中列名所在的行号：:violet[(手动译码数据为0，自动译码数据为4)]", "4")

# 创建输入框来获取要删除的行数
num_rows_to_skip_before = st.sidebar.number_input("要跳过的行数（前）：", min_value=0, value=0)
num_rows_to_skip_after = st.sidebar.number_input("要删除的行数（后）：", min_value=0, value=0)

# 将处理Time列空值的函数单独提取出来
def handle_time_column_na(data):
    if 'Time' in data.columns and data['Time'].isnull().any():
        st.warning(f"Time列中存在 {data['Time'].isnull().sum()} 个空值")
        time_na_handling = st.selectbox(
            "请选择Time列空值处理方法",
            ["删除空值行", "前向填充", "后向填充"]
        )
        
        if time_na_handling == "删除空值行":
            data = data.dropna(subset=['Time'])
            st.info(f"已删除Time列中的空值行，剩余 {len(data)} 行数据")
        elif time_na_handling == "前向填充":
            data['Time'].fillna(method='ffill', inplace=True)
            st.info("已用前一个有效值填充Time列的空值")
        elif time_na_handling == "后向填充":
            data['Time'].fillna(method='bfill', inplace=True)
            st.info("已用后一个有效值填充Time列的空值")
    return data

# 修改缓存函数，移除widget
@st.cache_data
def load_data(files, header, num_rows_to_skip_before, num_rows_to_skip_after):
    data_list = []
    date_list = []

    # 正则表达式模式，用于匹配 YYYYMMDD 格式
    date_pattern = re.compile(r'(\d{8})')

    for file in files:
        file_extension = file.name.split(".")[-1].lower()
        if file_extension == "csv":
            data = pd.read_csv(file, index_col=None, header=int(header), encoding='gb18030')
        elif file_extension == "xlsx":
            data = pd.read_excel(file, index_col=None, header=int(header))

        # 删除前后指定的行数
        if num_rows_to_skip_before > 0:
            data = data.iloc[num_rows_to_skip_before:]
        if num_rows_to_skip_after > 0:
            data = data.iloc[:-num_rows_to_skip_after]

        # 仅在上传多个文件时提取文件名中的日期信息
        if len(files) > 1:
            match = date_pattern.search(file.name)
            if match:
                date_str = match.group(1)  # YYYYMMDD
                file_date = pd.to_datetime(date_str, format='%Y%m%d')
                date_list.append(file_date)
            else:
                date_list.append(pd.NaT)
        
        data_list.append(data)

    # 合并多个文件
    merged_data = pd.concat(data_list, axis=0)
    return merged_data

# 导入数据
uploaded_files = st.file_uploader("📁 Please select the data files to import (multiple files can be uploaded):", type=["csv", "xlsx"],
                                  accept_multiple_files=True)
data = pd.DataFrame()  # 初始化为空的DataFrame

def process_time_column(data):
    try:
        if 'Time' in data.columns:
            # 首先检查第一行数据的格式
            sample_time = str(data['Time'].iloc[0])
            
            # 如果时间字符串包含日期部分
            if '/' in sample_time or '-' in sample_time:
                st.info("检测到Time列包含日期信息，正在处理...")
                # 尝试直接转换为datetime
                data['DateTime'] = pd.to_datetime(data['Time'], format='mixed')
                # 不需要额外的日期处理，因为日期信息已经包含在内
            else:
                # 处理纯时间格式 (HH:MM:SS)
                st.info("检测到Time列为纯时间格式，正在处理...")
                data['Time'] = pd.to_datetime(data['Time'], format='mixed').dt.time
                
                # 检测是否存在跨零点情况
                times = pd.to_datetime(data['Time'].astype(str))
                time_diff = times.diff()
                
                # 如果存在负的时间差，说明跨零点
                if (time_diff < pd.Timedelta(0)).any():
                    st.info("检测到时间序列跨零点，正在进行日期调整...")
                    
                    # 初始化日期偏移
                    date_offset = pd.Timedelta(days=0)
                    new_dates = []
                    
                    # 获取基准日期
                    if 'FileDate' in data.columns:
                        base_date = pd.to_datetime(data['FileDate'].iloc[0])
                    else:
                        base_date = pd.to_datetime('today')
                    
                    prev_time = None
                    for current_time in data['Time']:
                        if prev_time is not None:
                            # 如果当前时间小于前一个时间，说明跨零点
                            if current_time < prev_time:
                                date_offset += pd.Timedelta(days=1)
                        
                        # 将时间和日期偏移组合
                        new_datetime = base_date + date_offset + pd.Timedelta(
                            hours=current_time.hour,
                            minutes=current_time.minute,
                            seconds=current_time.second
                        )
                        new_dates.append(new_datetime)
                        prev_time = current_time
                    
                    data['DateTime'] = new_dates
                else:
                    # 如果没有跨零点，正常处理
                    if 'FileDate' in data.columns:
                        data['DateTime'] = data.apply(
                            lambda row: pd.to_datetime(str(row['FileDate'])) + pd.Timedelta(
                                hours=row['Time'].hour,
                                minutes=row['Time'].minute,
                                seconds=row['Time'].second
                            ),
                            axis=1
                        )
                    else:
                        today = pd.to_datetime('today')
                        data['DateTime'] = data['Time'].apply(
                            lambda x: today + pd.Timedelta(
                                hours=x.hour,
                                minutes=x.minute,
                                seconds=x.second
                            )
                        )

            # 设置DateTime为索引
            if 'DateTime' in data.columns:
                data.set_index('DateTime', inplace=True)
                data.sort_index(inplace=True)
                st.success("时间列处理成功！")
            
            return data
        else:
            st.warning("数据中没有找到Time列")
            return data
            
    except Exception as e:
        st.error(f"处理Time列时出错：{str(e)}")
        st.error("请检查Time列的格式是否一致")
        return None
    
if uploaded_files is not None and len(uploaded_files) > 0:
    # 数据加载
    data = load_data(uploaded_files, header, num_rows_to_skip_before, num_rows_to_skip_after)
    
    # 处理Time列的空值
    if data is not None and 'Time' in data.columns:
        # 处理空值
        data = handle_time_column_na(data)
        # 处理时间格式
        data = process_time_column(data)

    st.success("数据已成功导入并合并！")

    # 显示表格数据
    st.subheader("原始表格数据：")
    show_data = st.checkbox('是否显示原始表格数据的前20行', value=False)
    if show_data:
        st.dataframe(data.head(20))

    # 数据预处理部分
    columns = st.sidebar.multiselect("选择要预处理的列", data.columns.tolist())

    if columns:
        # 检查是否包含Time列
        if 'Time' in columns:
            st.info("检测到Time列，将自动将其设置为索引，并保持其时间格式")
            try:
                # 处理Time列中的空值
                if data['Time'].isnull().any():
                    st.warning(f"Time列中存在 {data['Time'].isnull().sum()} 个空值")
                    time_na_handling = st.selectbox(
                        "请选择Time列空值处理方法",
                        ["删除空值行", "前向填充", "后向填充"]
                    )
                    
                    if time_na_handling == "删除空值行":
                        data = data.dropna(subset=['Time'])
                        st.info(f"已删除Time列中的空值行，剩余 {len(data)} 行数据")
                    elif time_na_handling == "前向填充":
                        data['Time'].fillna(method='ffill', inplace=True)
                        st.info("已用前一个有效值填充Time列的空值")
                    elif time_na_handling == "后向填充":
                        data['Time'].fillna(method='bfill', inplace=True)
                        st.info("已用后一个有效值填充Time列的空值")

                # 将Time列转换为datetime.time类型
                data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S').dt.time

                # 将Time列与FileDate结合
                if 'FileDate' in data.columns:
                    data['DateTime'] = data.apply(lambda row: datetime.datetime.combine(row['FileDate'], row['Time']), axis=1)
                    data.set_index('DateTime', inplace=True)
                    columns.remove('Time')
                    st.success("Time列已成功设置为索引")
                
                # 显示时间范围信息
                time_range = data.index.max() - data.index.min()
                st.info(f"""
                数据时间范围信息：
                - 起始时间：{data.index.min().strftime('%Y-%m-%d %H:%M:%S')}
                - 结束时间：{data.index.max().strftime('%Y-%m-%d %H:%M:%S')}
                - 总时长：{time_range}
                - 数据点数：{len(data)}
                """)
                
            except Exception as e:
                st.error(f"Time列转换失败: {str(e)}")
                st.warning("Time列将保持原格式")
                columns.remove('Time')

        if columns:  # 确保还有其他列需要处理
            # 数据类型转换
            convert_types = st.selectbox("选择数据类型转换方式", ["不进行转换", "字符串转数值"])
            if convert_types == "字符串转数值":
                for col in columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                st.write("类型转换后的数据预览：", data[columns].head(10))

            # 空值处理
            missing_handling = st.selectbox("选择空值处理方法", ["不处理", "前向填充", "后向填充", "删除空值", "线性插值"])
            if missing_handling == "前向填充":
                data[columns] = data[columns].fillna(method='ffill')
            elif missing_handling == "后向填充":
                data[columns] = data[columns].fillna(method='bfill')
            elif missing_handling == "删除空值":
                data = data.dropna(subset=columns)
            elif missing_handling == "线性插值":
                # 进行线性插值
                data[columns] = data[columns].interpolate(method='linear')
                # 删除插值后仍然存在的空值
                data = data.dropna(subset=columns)
            st.write("空值处理后的数据预览：", data[columns].head(10))

            # 归一化
            normalization = st.selectbox("选择归一化方法", ["不进行归一化", "最小-最大归一化", "Z-score标准化", "最大绝对值归一化", "Robust Scaler", "L2归一化"])
            if normalization == "最小-最大归一化":
                scaler = MinMaxScaler()
                data[columns] = scaler.fit_transform(data[columns])
            elif normalization == "Z-score标准化":
                scaler = StandardScaler()
                data[columns] = scaler.fit_transform(data[columns])
            elif normalization == "最大绝对值归一化":
                scaler = MaxAbsScaler()
                data[columns] = scaler.fit_transform(data[columns])
            elif normalization == "Robust Scaler":
                scaler = RobustScaler()
                data[columns] = scaler.fit_transform(data[columns])
            elif normalization == "L2归一化":
                scaler = Normalizer(norm='l2')
                data[columns] = scaler.fit_transform(data[columns])
            st.write("归一化后的数据预览：", data[columns].head(10))

            # 主成分分析（PCA）
            apply_pca = st.checkbox("是否进行主成分分析（PCA）", value=False)
            if apply_pca:
                pca_columns = st.multiselect("选择要进行PCA的特征列", columns)
                if pca_columns:
                    n_components = st.number_input("选择主成分数量", min_value=1, max_value=len(pca_columns), value=2)
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(data[pca_columns])
                    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
                    st.write("PCA结果预览：", pca_df.head(10))

                    # 计算并展示累积方差图表
                    explained_variance_ratio = pca.explained_variance_ratio_
                    cumulative_variance_ratio = explained_variance_ratio.cumsum()

                    fig = px.line(
                        x=range(1, n_components + 1),
                        y=cumulative_variance_ratio,
                        labels={'x': '主成分数量', 'y': '累积方差比例'},
                        title='累积方差图表'
                    )
                    fig.update_traces(mode='markers+lines')
                    st.plotly_chart(fig)

                    # 输出每个主成分解释的方差比例
                    st.write("每个主成分解释的方差比例：")
                    for i, variance_ratio in enumerate(explained_variance_ratio):
                        st.write(f"主成分 {i+1}: {variance_ratio:.4f}")

                    # 提供下载选项
                    csv_buffer = io.BytesIO()
                    pca_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                    csv_buffer.seek(0)
                    st.download_button(
                        label="下载PCA结果",
                        data=csv_buffer,
                        file_name="PCA结果.csv",
                        mime="text/csv"
                    )

            # 预处理数据下载
            st.subheader("下载预处理后的数据")
            preprocessed_data = None

            if st.button("生成预处理数据"):
                if columns:
                    # 如果Time是索引，将其包含在导出数据中
                    if isinstance(data.index, pd.DatetimeIndex):
                        preprocessed_data = data[columns].copy()
                        preprocessed_data.index.name = 'Time'
                        preprocessed_data = preprocessed_data.reset_index()
                    else:
                        preprocessed_data = data[columns].copy()
                    st.success("预处理数据已生成，可以下载")
                else:
                    st.warning("请先选择要预处理的列")

            if preprocessed_data is not None:
                csv_buffer = io.BytesIO()
                preprocessed_data.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
                csv_buffer.seek(0)

                st.download_button(
                    label="下载预处理数据",
                    data=csv_buffer,
                    file_name="预处理数据.csv",
                    mime="text/csv"
                )

        # 确保数据按时间排序并设置时间索引
        if 'DateTime' in data.columns:
            data.sort_values(by='DateTime', inplace=True)
            data.set_index('DateTime', inplace=True)
            st.success("数据已按DateTime列排序并设置为索引")

        # 确保索引是DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            st.error("数据索引不是DatetimeIndex，请检查数据处理流程。")

        # 时间序列分解分析
        st.subheader("时间序列分解分析")

        # 选择要分析的列
        decompose_column = st.selectbox(
            "选择要进行时间序列分解的列",
            [""] + data.columns.tolist(),  # 在列名前添加一个空字符串
            index=0  # 设置默认选择为第一个选项，即空字符串
        )

        if decompose_column:
            # 添加模型选择和周期设置
            col1, col2 = st.columns(2)
            with col1:
                model = st.selectbox(
                    "选择分解模型",
                    ["加法模型", "乘法模型"]
                )
            with col2:
                period_type = st.selectbox(
                    "选择周期类型",
                    ["小时", "天", "周", "月"]
                )
                if period_type == "小时":
                    period = st.number_input("设置小时周期", min_value=1, value=24)
                elif period_type == "天":
                    period = st.number_input("设置天周期", min_value=1, value=7) * 24
                elif period_type == "周":
                    period = st.number_input("设置周周期", min_value=1, value=1) * 24 * 7
                else:  # 月
                    period = st.number_input("设置月周期", min_value=1, value=1) * 24 * 30

            # 添加生成图形的按钮
            if st.button("生成时间序列分解图"):
                try:
                    # 获取数据并确保按时间排序
                    ts_data = data[decompose_column].sort_index()

                    # 插入 NaN 来表示时间序列中的断点
                    ts_data = ts_data.asfreq('T')  # 假设数据是按分钟频率采样的
                    ts_data = ts_data.interpolate(method='time', limit_direction='both')

                    # 显示数据信息
                    st.info(f"""
                    数据时间范围：
                    开始时间：{ts_data.index.min()}
                    结束时间：{ts_data.index.max()}
                    数据点数量：{len(ts_data)}
                    """)

                    # 进行时间序列分解
                    decomposition = seasonal_decompose(
                        ts_data,
                        period=int(period),
                        model='additive' if model == "加法模型" else 'multiplicative'
                    )

                    # 创建图表，增加垂直间距
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=('原始数据', '趋势', '季节性', '残差'),
                        vertical_spacing=0.15,
                        shared_xaxes=True
                    )

                    # 添加原始数据，设置 connectgaps=False 以不连接空白处
                    fig.add_trace(
                        go.Scatter(
                            x=ts_data.index, 
                            y=ts_data.values, 
                            name='原始数据',
                            connectgaps=False,  # 不连接空白处
                            mode='lines',  # 使用线条模式
                        ),
                        row=1, col=1
                    )

                    # 添加趋势
                    fig.add_trace(
                        go.Scatter(
                            x=ts_data.index, 
                            y=decomposition.trend, 
                            name='趋势',
                            connectgaps=False,
                            mode='lines',
                        ),
                        row=2, col=1
                    )

                    # 添加季节性
                    fig.add_trace(
                        go.Scatter(
                            x=ts_data.index, 
                            y=decomposition.seasonal, 
                            name='季节性',
                            connectgaps=False,
                            mode='lines',
                        ),
                        row=3, col=1
                    )

                    # 添加残差
                    fig.add_trace(
                        go.Scatter(
                            x=ts_data.index, 
                            y=decomposition.resid, 
                            name='残差',
                            connectgaps=False,
                            mode='lines',
                        ),
                        row=4, col=1
                    )

                    # 更新布局，优化标签和间距
                    fig.update_layout(
                        height=800,
                        showlegend=False,
                        margin=dict(t=60, b=20, l=80, r=20),
                    )

                    # 只为最底部的子图显示x轴标签
                    fig.update_xaxes(title_text="", showticklabels=False, row=1)
                    fig.update_xaxes(title_text="", showticklabels=False, row=2)
                    fig.update_xaxes(title_text="", showticklabels=False, row=3)
                    fig.update_xaxes(title_text="时间", row=4)

                    # 更新每个子图的 Y 轴标题，调整位置
                    fig.update_yaxes(title_text="值", title_standoff=5, row=1, col=1)
                    fig.update_yaxes(title_text="趋势", title_standoff=5, row=2, col=1)
                    fig.update_yaxes(title_text="季节性", title_standoff=5, row=3, col=1)
                    fig.update_yaxes(title_text="残差", title_standoff=5, row=4, col=1)

                    # 更新子图标题的字体和位置
                    for i in range(len(fig.layout.annotations)):
                        fig.layout.annotations[i].update(
                            y=fig.layout.annotations[i].y + 0.03,  # 向上移动子图标题
                            font=dict(size=12)  # 调整字体大小
                        )

                    # 显示图表
                    st.plotly_chart(fig, use_container_width=True)

                    # 显示统计信息
                    st.subheader("分解结果统计信息")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("趋势统计")
                        st.write(decomposition.trend.describe())
                    
                    with col2:
                        st.write("季节性统计")
                        st.write(decomposition.seasonal.describe())
                    
                    with col3:
                        st.write("残差统计")
                        st.write(decomposition.resid.describe())
                    
                    # 提供下载分解结果的功能
                    decomp_df = pd.DataFrame({
                        '原始数据': ts_data,
                        '趋势': decomposition.trend,
                        '季节性': decomposition.seasonal,
                        '残差': decomposition.resid
                    })
                    
                    csv_buffer = io.BytesIO()
                    decomp_df.to_csv(csv_buffer, index=True, encoding="utf-8-sig")
                    csv_buffer.seek(0)
                    
                    st.download_button(
                        label="下载分解结果",
                        data=csv_buffer,
                        file_name=f"{decompose_column}_时间序列分解.csv",
                        mime="text/csv"
                    )
                    
                    # 添加解释性说明
                    st.info("""
                    时间序列分解说明：
                    
                    1. 趋势（Trend）：反映数据的长期变化方向
                    2. 季节性（Seasonal）：反映数据的周期性变化规律
                    3. 残差（Residual）：去除趋势和季节性后的随机波动
                    
                    加法模型：原始数据 = 趋势 + 季节性 + 残差
                    乘法模型：原始数据 = 趋势 × 季节性 × 残差
                    """)
                    
                    # 添加周期性分析结果
                    st.subheader("周期性分析")
                    seasonal_pattern = decomposition.seasonal[:period]
                    
                    # 创建周期模式图
                    fig_pattern = go.Figure()
                    
                    # 创建时间点列表
                    time_points = np.arange(int(period))
                    
                    # 添加周期模式，确保x和y的数据类型一致
                    fig_pattern.add_trace(go.Scatter(
                        x=time_points,  # 使用numpy数组
                        y=np.array(seasonal_pattern),  # 转换为numpy数组
                        mode='lines+markers',
                        name='季节性模式'
                    ))
                    
                    # 更新布局
                    fig_pattern.update_layout(
                        title=f'{period_type}周期模式',
                        xaxis_title=f'周期内时间点（总计{period}小时）',
                        yaxis_title='季节性成分值',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_pattern, use_container_width=True)
                    
                    # 添加周期特征统计
                    st.write("周期特征统计：")
                    st.write({
                        "周期长度": f"{period}小时",
                        "周期峰值": f"{seasonal_pattern.max():.3f}",
                        "周期谷值": f"{seasonal_pattern.min():.3f}",
                        "周期波动范围": f"{seasonal_pattern.max() - seasonal_pattern.min():.3f}"
                    })

                except Exception as e:
                    st.error(f"分解过程中出现错误：{str(e)}")
                    st.warning("请确保数据已正确设置时间索引，且选择了合适的分解周期。")
        
        # 相关性分析
        st.subheader("相关性分析")
        
        # 选择目标变量
        target_var = st.selectbox(
            "选择目标变量",
            [""] + data.columns.tolist()
        )
        
        if target_var:
            # 选择特征变量（多选）
            feature_vars = st.multiselect(
                "选择特征变量（可多选）",
                [col for col in data.columns if col != target_var]
            )
            
            if feature_vars:
                # 选择相关系数类型
                corr_method = st.radio(
                    "选择相关系数类型",
                    ["Pearson", "Spearman"],
                    horizontal=True
                )
                
                # 数据预处理
                corr_data = data[[target_var] + feature_vars].copy()
                
                # 转换为数值类型
                for col in corr_data.columns:
                    corr_data[col] = pd.to_numeric(corr_data[col], errors='coerce')
                
                # 使用线性插值填充空值
                corr_data = corr_data.interpolate(method='linear')
                # 处理首尾的空值
                corr_data = corr_data.fillna(method='bfill').fillna(method='ffill')
                
                if st.button("生成相关性热图"):
                    # 计算相关系数
                    corr_matrix = corr_data.corr(method=corr_method.lower())

                    # 创建热图
                    fig = ff.create_annotated_heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns.tolist(),
                        y=corr_matrix.index.tolist(),
                        colorscale='Viridis',
                        showscale=True,
                        annotation_text=corr_matrix.round(2).values,
                        hoverinfo="z"
                    )

                    fig.update_layout(
                        title=f'{corr_method} Correlation Matrix',
                        xaxis=dict(tickmode='array', tickvals=list(range(len(corr_matrix.columns))), ticktext=corr_matrix.columns),
                        yaxis=dict(tickmode='array', tickvals=list(range(len(corr_matrix.index))), ticktext=corr_matrix.index),
                        margin=dict(l=100, r=100, t=100, b=100)
                    )

                    # 显示图形
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示与目标变量的相关性排序
                    st.write(f"与{target_var}的{corr_method}相关系数排序：")
                    
                    # 获取目标变量的相关系数并排序
                    target_corr = corr_matrix[target_var].drop(target_var)
                    target_corr_sorted = target_corr.abs().sort_values(ascending=False)
                    
                    # 创建一个DataFrame来显示相关性排序
                    corr_df = pd.DataFrame({
                        '变量': target_corr_sorted.index,
                        '相关系数': target_corr[target_corr_sorted.index],
                        '绝对值': target_corr_sorted.values
                    })
                    
                    # 设置显示格式
                    pd.set_option('display.float_format', lambda x: '%.3f' % x)
                    
                    # 显示结果
                    st.dataframe(corr_df)
                    
                    # 添加解释性文本
                    st.info("""
                    相关系数解释：
                    - 取值范围：-1 到 1
                    - 1：完全正相关
                    - -1：完全负相关
                    - 0：无相关性
                    - 绝对值越大表示相关性越强
                    """)

        # 频谱分析部分
        st.subheader("频谱分析")

        # 选择要进行频谱分析的列
        spectrum_columns = st.multiselect("选择要进行频谱分析的列", data.columns.tolist(), key='spectrum_cols')

        # 选择筛选条件的列（可多选）
        filter_columns = st.multiselect(
            "选择用于筛选数据范围的列（可多选）",
            [col for col in data.columns if col not in spectrum_columns]  # 排除已选择的分析列
        )

        filter_conditions = []
        filter_ranges = {}

        if filter_columns:
            st.write("设置数据筛选范围：")
            for col in filter_columns:
                min_val = float(data[col].min())
                max_val = float(data[col].max())
                
                range_min, range_max = st.slider(
                    f"选择 {col} 的范围",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=(max_val - min_val) / 100
                )
                
                filter_ranges[col] = (range_min, range_max)
                filter_conditions.append(
                    (data[col] >= range_min) & (data[col] <= range_max)
                )

        # 应用筛选条件
        if filter_columns and filter_conditions:
            final_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                final_filter = final_filter & condition
            filtered_data = data[final_filter][spectrum_columns]
        else:
            filtered_data = data[spectrum_columns]

        if spectrum_columns:
            analysis_type = st.selectbox("选择分析方法", ["快速傅里叶变换(FFT)", "连续小波变换(CWT)"])
            
            if analysis_type == "快速傅里叶变换(FFT)":
                if st.button("生成FFT频谱图"):
                    # Calculate layout
                    n_cols = min(2, len(spectrum_columns))
                    n_rows = (len(spectrum_columns) + n_cols - 1) // n_cols
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3*n_rows))
                    if len(spectrum_columns) == 1:
                        axes = np.array([[axes]])
                    elif n_rows == 1:
                        axes = axes.reshape(1, -1)
                    
                    for idx, col in enumerate(spectrum_columns):
                        row = idx // n_cols
                        col_idx = idx % n_cols
                        
                        signal = filtered_data[col].values
                        n = len(signal)
                        fft_result = np.fft.fft(signal)
                        freq = np.fft.fftfreq(n, d=1)
                        
                        axes[row, col_idx].plot(freq[:n//2], np.abs(fft_result)[:n//2])
                        axes[row, col_idx].set_title(f'FFT of {col}')
                        axes[row, col_idx].set_xlabel('Frequency (Hz)')
                        axes[row, col_idx].set_ylabel('Magnitude')
                        axes[row, col_idx].grid(True)
                    
                    # Handle extra subplots
                    for idx in range(len(spectrum_columns), n_rows * n_cols):
                        row = idx // n_cols
                        col_idx = idx % n_cols
                        fig.delaxes(axes[row, col_idx])
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            
            else:  # CWT分析
                if st.button("生成CWT时频图"):
                    import pywt
                    scales = np.arange(1, 128)
                    wavelet = 'morl'
                    
                    for col in spectrum_columns:
                        signal = filtered_data[col].values
                        coefficients, frequencies = pywt.cwt(signal, scales, wavelet)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 2D Time-Frequency plot
                            fig, ax = plt.subplots(figsize=(5, 3))
                            im = ax.pcolormesh(np.arange(len(signal)), frequencies, np.abs(coefficients))
                            ax.set_title(f'CWT of {col}')
                            ax.set_ylabel('Frequency (Hz)')
                            ax.set_xlabel('Time (s)')
                            cbar = plt.colorbar(im, ax=ax, label='Magnitude')
                            # 调整colorbar的字体大小
                            cbar.ax.tick_params(labelsize=8)
                            # 调整主图的字体大小
                            ax.tick_params(labelsize=8)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                        with col2:
                            # 3D Time-Frequency plot
                            fig = plt.figure(figsize=(5, 3))
                            ax = fig.add_subplot(111, projection='3d')
                            time_grid, scale_grid = np.meshgrid(np.arange(len(signal)), frequencies)
                            surf = ax.plot_surface(time_grid, scale_grid, np.abs(coefficients), cmap='viridis', linewidth=0)
                            ax.set_title(f'3D CWT of {col}', pad=2)  # 减小标题和图之间的间距
                            ax.set_xlabel('Time (s)', labelpad=2)    # 减小标签和轴之间的间距
                            ax.set_ylabel('Frequency (Hz)', labelpad=2)
                            ax.set_zlabel('Magnitude', labelpad=2)
                            # 调整视角
                            ax.view_init(elev=30, azim=45)
                            # 调整轴标签字体大小
                            ax.tick_params(labelsize=8)
                            cbar = plt.colorbar(surf, ax=ax, label='Magnitude', pad=0.1)  # 减小colorbar的间距
                            cbar.ax.tick_params(labelsize=8)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

        # 概率密度分布分析部分
        st.subheader("概率密度分布分析")
        
        # 多选
        density_columns = st.multiselect(
            "选择要分析概率密度分布的列（可多选）",
            data.columns.tolist(),
            key='density_cols'  # 添加唯一的key
        )
        
        if density_columns:
            # 选择筛选条件的列（可多选）
            filter_columns = st.multiselect(
                "选择用于筛选数据范围的列（可多选）",
                [col for col in data.columns if col not in density_columns],  # 排除已选择的分析列
                key='filter_cols'  # 添加唯一的key
            )
            
            if filter_columns:
                # 数据预处理
                processed_data = data.copy()
                
                for col in filter_columns:
                    processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')
                    processed_data[col] = processed_data[col].interpolate(method='linear')
                    processed_data[col] = processed_data[col].fillna(method='bfill').fillna(method='ffill')
                
                st.info(f"数据预处理完成：\n" + 
                       "\n".join([f"{col}: {processed_data[col].isna().sum()} 个空值" 
                                 for col in filter_columns]))
                
                filter_conditions = []
                filter_ranges = {}
                
                st.write("设置数据筛选范围：")
                for col in filter_columns:
                    min_val = float(processed_data[col].min())
                    max_val = float(processed_data[col].max())
                    
                    range_min, range_max = st.slider(
                        f"选择 {col} 的范围",
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val),
                        step=(max_val - min_val) / 100,
                        key=f"slider_{col}"  # 为每个slider添加唯一的key
                    )
                    
                    filter_ranges[col] = (range_min, range_max)
                    filter_conditions.append(
                        (processed_data[col] >= range_min) & (processed_data[col] <= range_max)
                    )
            
            if st.button("生成概率密度分布图"):
                # 应用筛选条件
                if filter_columns and filter_conditions:
                    final_filter = filter_conditions[0]
                    for condition in filter_conditions[1:]:
                        final_filter = final_filter & condition
                    filtered_data = processed_data[final_filter][density_columns]
                else:
                    filtered_data = data[density_columns]

                # 检查filtered_data是否为空或density_columns是否为空
                if filtered_data.empty or not density_columns:
                    st.warning("筛选后的数据为空或未选择任何列，请调整筛选条件。")
                else:
                    # 创建一个图表来显示所有选中列的概率密度分布
                    fig = go.Figure()
                    
                    # 为每个选的列添加直方图和密度曲线
                    for column in density_columns:
                        if column in filtered_data.columns:
                            # 添加直方图
                            fig.add_trace(go.Histogram(
                                x=filtered_data[column],
                                name=f'{column} - 直方图',
                                histnorm='probability density',
                                opacity=0.5,
                                nbinsx=50,
                                showlegend=True
                            ))
                            
                            # 添加核密度估计曲线
                            from scipy import stats
                            kde = stats.gaussian_kde(filtered_data[column].dropna())
                            x_range = np.linspace(filtered_data[column].min(), filtered_data[column].max(), 200)
                            fig.add_trace(go.Scatter(
                                x=x_range,
                                y=kde(x_range),
                                name=f'{column} - 密度曲线',
                                line=dict(width=2)
                            ))
                    
                    # 更新布局
                    fig.update_layout(
                        title="多参数概率密度分布对比图",
                        xaxis_title="值",
                        yaxis_title="密度",
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        ),
                        bargap=0.1,
                        barmode='overlay'  # 使直方图重叠显示
                    )
                    
                    # 显示图表
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示所有参数的基本统计信息
                    st.write("**所有参数的基本统计信息：**")
                    stats_df = pd.DataFrame({
                        '统计量': ['均值', '中位数', '标准差', '最小值', '最大值']
                    })
                    
                    for column in density_columns:
                        if column in filtered_data.columns:
                            stats_df[column] = [
                                f"{filtered_data[column].mean():.3f}",
                                f"{filtered_data[column].median():.3f}",
                                f"{filtered_data[column].std():.3f}",
                                f"{filtered_data[column].min():.3f}",
                                f"{filtered_data[column].max():.3f}"
                            ]
                    
                    st.write(stats_df)

        # 首先定义辅助函数
        def generate_future_features(X, future_dates, feature_columns):
            """
            根据历史数据生成未来特征
            这里需要根据实际特征的性质来实现具体的生成逻辑
            """
            # 示例实现：使用最后一天的特征值重复
            last_features = X.iloc[-1]
            future_features = pd.DataFrame([last_features.values] * len(future_dates),
                                         index=future_dates,
                                         columns=feature_columns)
            return future_features

        # 模型训练部分
        st.subheader("模型训练")
        
        # 添加空选项作为默认值
        target_options = [""] + data.columns.tolist()
        target_column = st.selectbox("选择目标列（预测目标）：", target_options)
        
        # 只有当选择了目标列时才显示后续选项
        if target_column:
            feature_columns = st.multiselect("选择特征列：", [col for col in data.columns if col != target_column])

            model_choice = st.selectbox("选择模型", ["线性回归", "多项式回归", "ARIMA", "梯度下降", "LSTM", "GRU", "XGBoost", "RandomForest", "LightGBM", "SVR"])

            # 添加模型参数设置
            if model_choice == "多项式回归":
                degree = st.number_input("多项式回归的阶数", min_value=2, max_value=5, value=2)
            elif model_choice == "ARIMA":
                p = st.number_input("ARIMA的p参数", min_value=0, max_value=5, value=1)
                d = st.number_input("ARIMA的d参数", min_value=0, max_value=5, value=1)
                q = st.number_input("ARIMA的q参数", min_value=0, max_value=5, value=1)
            elif model_choice == "梯度下降":
                learning_rate = st.slider("选择学习率", min_value=0.0001, max_value=1.0, value=0.01)
                max_iter = st.number_input("最大迭代次数", min_value=100, max_value=10000, value=1000)
                tol = st.number_input("容忍度", min_value=1e-3, max_value=1.0, value=1e-2)
                random_state = st.number_input("随机种子", min_value=0, max_value=1000, value=42)
            elif model_choice == "LSTM":
                # LSTM参数设置
                hidden_size = st.number_input("隐藏层大小", min_value=32, max_value=256, value=64)
                num_layers = st.number_input("LSTM层数", min_value=1, max_value=3, value=1)
                epochs = st.number_input("训练轮数", min_value=10, max_value=500, value=100)
                batch_size = st.number_input("批次大小", min_value=16, max_value=128, value=32)
                sequence_length = st.number_input("序列长度", min_value=1, max_value=48, value=24)
                learning_rate = st.slider("学习率", min_value=0.00001, max_value=0.01, value=0.001, step=0.0001, format="%.5f")
            elif model_choice == "GRU":
                # GRU参数设置
                hidden_size = st.number_input("隐藏层大小", min_value=32, max_value=256, value=64)
                num_layers = st.number_input("GRU层数", min_value=1, max_value=3, value=1)
                epochs = st.number_input("训练轮数", min_value=10, max_value=500, value=100)
                batch_size = st.number_input("批次大小", min_value=16, max_value=128, value=32)
                sequence_length = st.number_input("序列长度", min_value=1, max_value=48, value=24)
                learning_rate = st.slider("学习率", min_value=0.00001, max_value=0.01, value=0.001, step=0.0001, format="%.5f")
            elif model_choice == "XGBoost":
                n_estimators = st.number_input("树的数量", min_value=100, max_value=1000, value=100)
                max_depth = st.number_input("树的最大深度", min_value=1, max_value=10, value=3)
                learning_rate = st.slider("学习率", min_value=0.01, max_value=0.3, value=0.1)
                subsample = st.slider("子样本比例", min_value=0.5, max_value=1.0, value=1.0)
                colsample_bytree = st.slider("每棵树使用的特征比例", min_value=0.5, max_value=1.0, value=1.0)
                reg_lambda = st.number_input("L2正则化项", min_value=0.0, max_value=10.0, value=1.0)
                reg_alpha = st.number_input("L1正则化项", min_value=0.0, max_value=10.0, value=0.0)
            elif model_choice == "RandomForest":
                n_estimators = st.number_input("树的数量", min_value=100, max_value=1000, value=100)
                max_depth = st.number_input("树的最大深度", min_value=1, max_value=30, value=10)
                min_samples_split = st.number_input("内部节点再划分所需最小样本数", min_value=2, max_value=10, value=2)
                min_samples_leaf = st.number_input("叶子节点最小样本数", min_value=1, max_value=10, value=1)
                max_features = st.selectbox("最大特征数", ["指定浮点数值", "sqrt", "log2"])
                if max_features == "指定浮点数值":
                    max_features = st.number_input("输入最大特征数的比例 (0.0, 1.0]", min_value=0.0, max_value=1.0, value=0.5)
                else:
                    max_features = max_features
            elif model_choice == "LightGBM":
                n_estimators = st.number_input("树的数量", min_value=100, max_value=1000, value=100)
                max_depth = st.number_input("树的最大深度", min_value=-1, max_value=50, value=-1)
                learning_rate = st.slider("学习率", min_value=0.01, max_value=0.3, value=0.1)
                num_leaves = st.number_input("叶子节点数", min_value=2, max_value=256, value=31)
                min_child_samples = st.number_input("子节点最小样本数", min_value=1, max_value=100, value=20)
                metric = st.selectbox("选择评估指标", ["l2", "l1", "huber", "fair", "poisson", "quantile", "mape", "gamma", "tweedie"])
                boosting_type = st.selectbox("选择提升类型", ["gbdt", "dart", "goss", "rf"])
            elif model_choice == "SVR":
                kernel = st.selectbox("选择核函数", ["linear", "poly", "rbf", "sigmoid"])
                C = st.number_input("正则化参数C", min_value=0.1, max_value=100.0, value=1.0)
                epsilon = st.number_input("epsilon参数", min_value=0.01, max_value=1.0, value=0.1)

            # 添加未来预测天数选择
            future_days = st.number_input("预测未来天数", min_value=1, max_value=30, value=1)

            if st.button("开始训练模型"):
                try:
                    X = data[feature_columns]
                    y = data[target_column]
        
                    # 按时间顺序排序
                    data = data.sort_index()
                    X = X.sort_index()
                    y = y.sort_index()
        
                    # 保持索引的切分
                    train_size = int(len(data) * 0.8)
                    train_index = data.index[:train_size]
                    test_index = data.index[train_size:]
                    
                    X_train = X.loc[train_index]
                    X_test = X.loc[test_index]
                    y_train = y.loc[train_index]
                    y_test = y.loc[test_index]
        
                    # 生成未来时间点
                    last_date = data.index[-1]
                    future_dates = pd.date_range(start=last_date, periods=future_days*24+1, freq='H')[1:]
        
                    # 训练模型并预测
                    if model_choice == "线性回归":
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_train_pred = pd.Series(model.predict(X_train), index=train_index)
                        y_test_pred = pd.Series(model.predict(X_test), index=test_index)
                        
                        # 生成未来预测
                        future_X = generate_future_features(X, future_dates, feature_columns)
                        future_pred = pd.Series(model.predict(future_X), index=future_dates)
        
                    elif model_choice == "多项式回归":
                        poly = PolynomialFeatures(degree=degree)
                        X_poly_train = poly.fit_transform(X_train)
                        X_poly_test = poly.transform(X_test)
                        model = LinearRegression()
                        model.fit(X_poly_train, y_train)
                        y_train_pred = pd.Series(model.predict(X_poly_train), index=train_index)
                        y_test_pred = pd.Series(model.predict(X_poly_test), index=test_index)
                        
                        # 生成未来预测
                        future_X = generate_future_features(X, future_dates, feature_columns)
                        future_X_poly = poly.transform(future_X)
                        future_pred = pd.Series(model.predict(future_X_poly), index=future_dates)
        
                    elif model_choice == "ARIMA":
                        # 检查特征参数数量
                        if len(feature_columns) > 1:
                            st.warning("ARIMA模型只能处理单变量时间序列数据，请确保特征列数量为1。")
                        
                        # 确保索引是时间序列
                        y_train = y_train.asfreq('H')
                        y_test = y_test.asfreq('H')
        
                        model = sm.tsa.ARIMA(y_train, order=(p, d, q))
                        model_fit = model.fit()
        
                        # 获取训练集预测值
                        y_train_pred = pd.Series(model_fit.predict(start=train_index[0], end=train_index[-1]), index=train_index)
        
                        # 获取测试集预测值
                        y_test_pred = pd.Series(model_fit.predict(start=test_index[0], end=test_index[-1]), index=test_index)
        
                        # 预测未来值
                        future_steps = len(future_dates)
                        future_pred = pd.Series(model_fit.forecast(steps=future_steps), index=future_dates)
        
                    elif model_choice == "梯度下降":
                        model = SGDRegressor(
                            learning_rate='constant',
                            eta0=learning_rate,
                            max_iter=max_iter,
                            tol=tol,
                            random_state=random_state
                        )
                        model.fit(X_train, y_train)
                        y_train_pred = pd.Series(model.predict(X_train), index=train_index)
                        y_test_pred = pd.Series(model.predict(X_test), index=test_index)
                        
                        # 生成未来预测
                        future_X = generate_future_features(X, future_dates, feature_columns)
                        future_pred = pd.Series(model.predict(future_X), index=future_dates)

                    elif model_choice == "LSTM":
                        # 数据标准化
                        scaler_X = StandardScaler()
                        scaler_y = StandardScaler()
                        
                        X_scaled = scaler_X.fit_transform(X)
                        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
                        
                        # 准备训练集和测试集
                        train_data = TimeSeriesDataset(
                            X_scaled[:train_size], 
                            y_scaled[:train_size], 
                            sequence_length
                        )
                        test_data = TimeSeriesDataset(
                            X_scaled[train_size:], 
                            y_scaled[train_size:], 
                            sequence_length
                        )
                        
                        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                        
                        # 初始化模型
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = LSTMModel(len(feature_columns), hidden_size, num_layers).to(device)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                        # 初始化一个空列表来存储每个epoch的训练损失
                        train_losses = []
                        
                        # 训练模型
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for epoch in range(epochs):
                            model.train()
                            running_loss = 0.0
                            for batch_X, batch_y in train_loader:
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                optimizer.zero_grad()
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                                loss.backward()
                                optimizer.step()

                                running_loss += loss.item()

                            # 计算并记录每个epoch的平均训练损失
                            epoch_loss = running_loss / len(train_loader)
                            train_losses.append(epoch_loss)
                            
                            progress_bar.progress((epoch + 1) / epochs)
                            status_text.text(f'训练进度: {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

                        # 绘制训练损失图
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(x=np.arange(1, epochs+1), y=train_losses, mode='lines+markers'))
                        fig_loss.update_layout(title='训练损失曲线', xaxis_title='Epoch', yaxis_title='Loss')
                        st.plotly_chart(fig_loss, use_container_width=True)
                        status_text.text('训练完成')
                        progress_bar.empty()
                        st.write('')

                        
                        # 生成预测
                        model.eval()
                        with torch.no_grad():
                            # 训练集预测
                            train_predictions = []
                            for batch_X, _ in train_loader:
                                batch_X = batch_X.to(device)
                                outputs = model(batch_X)
                                train_predictions.extend(outputs.cpu().numpy())
                            
                            # 测试集预测
                            test_predictions = []
                            for batch_X, _ in test_loader:
                                batch_X = batch_X.to(device)
                                outputs = model(batch_X)
                                test_predictions.extend(outputs.cpu().numpy())
                            
                            # 未来预测
                            future_X = generate_future_features(X, future_dates, feature_columns)
                            future_X_scaled = scaler_X.transform(future_X)
                            
                            # 准备最后一个序列用于预测
                            last_sequence = torch.FloatTensor(X_scaled[-sequence_length:]).unsqueeze(0).to(device)
                            future_predictions = []
                            
                            # 逐步预测未来值
                            for _ in range(len(future_dates)):
                                next_pred = model(last_sequence)
                                future_predictions.append(next_pred.item())
                                # 更新序列
                                last_sequence = torch.cat([
                                    last_sequence[:, 1:, :],
                                    torch.FloatTensor(future_X_scaled[_:_+1]).unsqueeze(0).to(device)
                                ], dim=1)
                            
                            # 转换回原始比例
                            y_train_pred = pd.Series(scaler_y.inverse_transform(np.array(train_predictions).reshape(-1, 1)).flatten(),index=train_index[sequence_length:])
                            y_test_pred = pd.Series(scaler_y.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten(),index=test_index[sequence_length:])
                            future_pred = pd.Series(scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten(),index=future_dates)
                            
                            # 调整评估数据的范围
                            y_train = y_train.iloc[sequence_length:]
                            y_test = y_test.iloc[sequence_length:]
        
                            # 使用统一评估函数计算指标
                            train_metrics = evaluate_model(y_train, y_train_pred)
                            test_metrics = evaluate_model(y_test, y_test_pred)

                            # 调用 display_metrics 函数显示评估指标
                            display_metrics(train_metrics, test_metrics)

                    elif model_choice == "GRU":
                        # 数据标准化
                        scaler_X = StandardScaler()
                        scaler_y = StandardScaler()
                        
                        X_scaled = scaler_X.fit_transform(X)
                        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
                        
                        # 准备训练集和测试集
                        train_data = TimeSeriesDataset(
                            X_scaled[:train_size], 
                            y_scaled[:train_size], 
                            sequence_length
                        )
                        test_data = TimeSeriesDataset(
                            X_scaled[train_size:], 
                            y_scaled[train_size:], 
                            sequence_length
                        )
                        
                        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
                        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                        
                        # 初始化模型
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = GRUModel(len(feature_columns), hidden_size, num_layers).to(device)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                        # 初始化一个空列表来存储每个epoch的训练损失
                        train_losses = []
                        
                        # 训练模型
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for epoch in range(epochs):
                            model.train()
                            running_loss = 0.0
                            for batch_X, batch_y in train_loader:
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                optimizer.zero_grad()
                                outputs = model(batch_X)
                                loss = criterion(outputs, batch_y)
                                loss.backward()
                                optimizer.step()

                                running_loss += loss.item()

                            # 计算并记录每个epoch的平均训练损失
                            epoch_loss = running_loss / len(train_loader)
                            train_losses.append(epoch_loss)
                            
                            progress_bar.progress((epoch + 1) / epochs)
                            status_text.text(f'训练进度: {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

                        # 绘制训练损失图
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(x=np.arange(1, epochs+1), y=train_losses, mode='lines+markers'))
                        fig_loss.update_layout(title='训练损失曲线', xaxis_title='Epoch', yaxis_title='Loss')
                        st.plotly_chart(fig_loss, use_container_width=True)
                        status_text.text('训练完成')
                        progress_bar.empty()

                        # 生成预测值
                        model.eval()
                        with torch.no_grad():
                            # 训练集预测
                            train_predictions = []
                            for batch_X, _ in train_loader:
                                batch_X = batch_X.to(device)
                                outputs = model(batch_X)
                                train_predictions.extend(outputs.cpu().numpy())
                            
                            # 测试集预测
                            test_predictions = []
                            for batch_X, _ in test_loader:
                                batch_X = batch_X.to(device)
                                outputs = model(batch_X)
                                test_predictions.extend(outputs.cpu().numpy())
                            
                            # 未来预测
                            future_X = generate_future_features(X, future_dates, feature_columns)
                            future_X_scaled = scaler_X.transform(future_X)
                            
                            # 准备最后一个序列用于预测
                            last_sequence = torch.FloatTensor(X_scaled[-sequence_length:]).unsqueeze(0).to(device)
                            future_predictions = []
                            
                            # 逐步预测未来值
                            for _ in range(len(future_dates)):
                                next_pred = model(last_sequence)
                                future_predictions.append(next_pred.item())
                                # 更新序列
                                last_sequence = torch.cat([
                                    last_sequence[:, 1:, :],
                                    torch.FloatTensor(future_X_scaled[_:_+1]).unsqueeze(0).to(device)
                                ], dim=1)
                            
                            # 转换回原始比例
                            y_train_pred = pd.Series(scaler_y.inverse_transform(np.array(train_predictions).reshape(-1, 1)).flatten(),index=train_index[sequence_length:])
                            y_test_pred = pd.Series(scaler_y.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten(),index=test_index[sequence_length:])
                            future_pred = pd.Series(scaler_y.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten(),index=future_dates)
                            
                            # 调整评估数据的范围
                            y_train = y_train.iloc[sequence_length:]
                            y_test = y_test.iloc[sequence_length:]
        
                            # 使用统一评估函数计算指标
                            train_metrics = evaluate_model(y_train, y_train_pred)
                            test_metrics = evaluate_model(y_test, y_test_pred)

                            # 调用 display_metrics 函数显示评估指标
                            display_metrics(train_metrics, test_metrics)                

                    elif model_choice == "XGBoost":
                        import xgboost as xgb
                        model = xgb.XGBRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            reg_lambda=reg_lambda,
                            reg_alpha=reg_alpha
                        )
                        model.fit(X_train, y_train)
                        y_train_pred = pd.Series(model.predict(X_train), index=train_index)
                        y_test_pred = pd.Series(model.predict(X_test), index=test_index)
                        
                        shap_analysis(model, X_train, X_test, feature_columns)
                        # 使用统一评估函数计算指标
                        train_metrics = evaluate_model(y_train, y_train_pred)
                        test_metrics = evaluate_model(y_test, y_test_pred)

                        # 调用 display_metrics 函数显示评估指标
                        display_metrics(train_metrics, test_metrics)
                        
                        # 生成未来预测
                        future_X = generate_future_features(X, future_dates, feature_columns)
                        future_pred = pd.Series(model.predict(future_X), index=future_dates)

                    elif model_choice == "RandomForest":
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features
                        )
                        model.fit(X_train, y_train)
                        y_train_pred = pd.Series(model.predict(X_train), index=train_index)
                        y_test_pred = pd.Series(model.predict(X_test), index=test_index)

                        shap_analysis(model, X_train, X_test, feature_columns)
                        # 使用统一评估函数计算指标
                        train_metrics = evaluate_model(y_train, y_train_pred)
                        test_metrics = evaluate_model(y_test, y_test_pred)

                        # 调用 display_metrics 函数显示评估指标
                        display_metrics(train_metrics, test_metrics)
                        
                        # 生成未来预测
                        future_X = generate_future_features(X, future_dates, feature_columns)
                        future_pred = pd.Series(model.predict(future_X), index=future_dates)

                    elif model_choice == "LightGBM":
                        model = lgb.LGBMRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            num_leaves=num_leaves,
                            min_child_samples=min_child_samples,
                            metric=metric,
                            boosting_type=boosting_type
                        )
                        model.fit(X_train, y_train)
                        y_train_pred = pd.Series(model.predict(X_train), index=train_index)
                        y_test_pred = pd.Series(model.predict(X_test), index=test_index)

                        shap_analysis(model, X_train, X_test, feature_columns)
                        # 使用统一评估函数计算指标
                        train_metrics = evaluate_model(y_train, y_train_pred)
                        test_metrics = evaluate_model(y_test, y_test_pred)

                        # 调用 display_metrics 函数显示评估指标
                        display_metrics(train_metrics, test_metrics)
                        
                        # 生成未来预测
                        future_X = generate_future_features(X, future_dates, feature_columns)
                        future_pred = pd.Series(model.predict(future_X), index=future_dates)
                        
                    elif model_choice == "SVR":
                        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
                        model.fit(X_train, y_train)
                        y_train_pred = pd.Series(model.predict(X_train), index=train_index)
                        y_test_pred = pd.Series(model.predict(X_test), index=test_index)

                        shap_analysis(model, X_train, X_test, feature_columns)                       
                        # 生成未来预测
                        future_X = generate_future_features(X, future_dates, feature_columns)
                        future_pred = pd.Series(model.predict(future_X), index=future_dates)

                        # 使用统一评估函数计算指标
                        train_metrics = evaluate_model(y_train, y_train_pred)
                        test_metrics = evaluate_model(y_test, y_test_pred)

                        # 调用 display_metrics 函数显示评估指标
                        display_metrics(train_metrics, test_metrics)
        
                    # 创建单个图表替代原来的三个子图
                    fig = go.Figure()
        
                    # 添加训练集数据
                    fig.add_trace(
                        go.Scatter(
                            x=y_train.index,
                            y=y_train.values,
                            name='训练集实际值',
                            mode='lines',
                            line=dict(color='blue')
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=y_train_pred.index,
                            y=y_train_pred.values,
                            name='训练集预测值',
                            mode='lines',
                            line=dict(color='red', dash='dash')
                        )
                    )
        
                    # 添加测试集数据
                    fig.add_trace(
                        go.Scatter(
                            x=y_test.index,
                            y=y_test.values,
                            name='测试集实际值',
                            mode='lines',
                            line=dict(color='green')
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=y_test_pred.index,
                            y=y_test_pred.values,
                            name='测试集预测值',
                            mode='lines',
                            line=dict(color='orange', dash='dash')
                        )
                    )
        
                    # 添加未来预测数据
                    fig.add_trace(
                        go.Scatter(
                            x=future_dates,
                            y=future_pred.values,
                            name='未来预测',
                            mode='lines',
                            line=dict(color='purple')
                        )
                    )
        
                    # 更新布局
                    fig.update_layout(
                        height=600,
                        title=f"{model_choice}模型预测结果",
                        xaxis_title="时间",
                        yaxis_title=target_column,
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                        # 添加垂直分隔线来区分训练集、测试集和未来预测
                        shapes=[
                            # 训练集和测试集的分隔线
                            dict(
                                type="line",
                                x0=test_index[0],
                                x1=test_index[0],
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(color="gray", width=1, dash="dash")
                            ),
                            # 测试集和未来预测的分隔线
                            dict(
                                type="line",
                                x0=future_dates[0],
                                x1=future_dates[0],
                                y0=0,
                                y1=1,
                                yref="paper",
                                line=dict(color="gray", width=1, dash="dash")
                            )
                        ],
                        # 添加注释
                        annotations=[
                            dict(
                                x=y_train.index[len(y_train)//2],
                                y=1.05,
                                yref="paper",
                                text="训练集",
                                showarrow=False
                            ),
                            dict(
                                x=y_test.index[len(y_test)//2],
                                y=1.05,
                                yref="paper",
                                text="测试集",
                                showarrow=False
                            ),
                            dict(
                                x=future_dates[len(future_dates)//2],
                                y=1.05,
                                yref="paper",
                                text="未来预测",
                                showarrow=False
                            )
                        ]
                    )
        
                    # 显示图表
                    st.plotly_chart(fig, use_container_width=True)
        
                except Exception as e:
                    st.error(f"模型训练过程中出现错误：{str(e)}")
                    st.warning("请检查数据格式和选择的特征是否合适。")
                    

# 在主要代码的最后添加侧边栏信息

st.sidebar.markdown("---")

# 添加版本记录
with st.sidebar.expander("版本记录", expanded=True, icon="🚨"):
    st.markdown("""
    **1.1** 改版内容：增加时间序列分解分析模块。  
    **1.2** 改版内容：增加频谱分析的数据筛选选项。  
    **1.3** 改版内容：多文档上传后的日期信息提取；优化线性插值后的空值处理。  
    **1.4** 改版内容：模型训练模块添加了未来预测天数的选择。  
    **1.5** 改版内容：部分图表使用plotly优化曲线显示效果。  
    **1.6** 改版内容：模型训练增加LSTM算法。  
    **1.7** 改版内容：代码结构化，便于维护。  
    **1.8** 改版内容：代码回退至V1.6版本，并解决添加了时序数据跨零点检测逻辑（当检测到跨零点时，自动添加日期偏移）。同时优化了模型训练后的可视化显示。  
    **1.9** 改版内容：增加XGBoost模型训练算法。  
    **2.0** 改版内容：增加RandomForest模型训练算法。  
    **2.1** 改版内容：增加LightGBM模型训练算法。  
    **2.2** 改版内容：增加SVR模型训练算法。  
    **2.3** 改版内容：添加统一的模型评估函数使所有模型（LSTM/XGBoost/RandomForest/LightGBM/SVR）都有相同的评估指标。  
    **2.4** 改版内容：增加主成分分析（PCA），优化相关性热图显示。   
    **2.5** 改版内容：增加LSTM模型训练过程中的训练损失曲线显示。   
    **2.6** 改版内容：增加GRU模型训练算法和归一化方法（最大绝对值归一化、Robust Scaler和L2归一化）。   
    **2.7** 改版内容：增加SHAP值分析功能，优化评估指标显示。   
    """)
        
st.sidebar.markdown("---")
        
# 添加开发者信息
st.sidebar.markdown("""
### 开发者信息
        
**开发者**：王康业  
**邮箱**：kangy_wang@hnair.com
        
---
        
### 版权声明
        
Copyright © 2025 王康业. All Rights Reserved.  
                            
本应用程序受著作权法和其他知识产权法保护。  
未经授权，禁止复制、修改或分发本程序的任何部分。
                    
Version 2.7.0
""")
        
# 添加一些空行来确保版权信息在底部
st.sidebar.markdown("<br>" * 5, unsafe_allow_html=True)
