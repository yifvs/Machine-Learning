import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import numpy as np
import io
import matplotlib.pyplot as plt
import datetime
import seaborn as sns


# 设置页面布局
st.set_page_config(layout="wide", page_title="Machine Learning", page_icon="📈")
# 设置应用标题
st.title("数据预处理与模型训练 Web 应用")

# 创建一个输入框来获取header的值
header = st.sidebar.text_input("请输入数据表中列名所在的行号：:violet[(手动译码数据为0，自动译码数据为4)]", "4")

# 创建输入框来获取要删除的行数
num_rows_to_skip_before = st.sidebar.number_input("要跳过的行数（前）：", min_value=0, value=0)
num_rows_to_skip_after = st.sidebar.number_input("要删除的行数（后）：", min_value=0, value=0)


# 缓存加载数据的函数，支持多个文件上和合并
@st.cache_data
def load_data(files, header, num_rows_to_skip_before, num_rows_to_skip_after):
    data_list = []

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

        data_list.append(data)

    # 合并多个文件
    merged_data = pd.concat(data_list, axis=0)
    return merged_data


# 导入数据
uploaded_files = st.file_uploader("📁 Please select the data files to import (multiple files can be uploaded):", type=["csv", "xlsx"],
                                  accept_multiple_files=True)
data = pd.DataFrame()  # 初始化为空的DataFrame

if uploaded_files is not None and len(uploaded_files) > 0:
    # 数据加载部分
    data = load_data(uploaded_files, header, num_rows_to_skip_before, num_rows_to_skip_after)
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
                # 首先处理Time列中的空值
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

                # 检查第一个非空值的类型并相应处理
                first_valid_time = data['Time'].dropna().iloc[0]
                
                if isinstance(first_valid_time, str):
                    # 如果是字符串，尝试直接转换
                    data['Time'] = pd.to_datetime(data['Time'], format='mixed')
                elif isinstance(first_valid_time, datetime.time):
                    # 如果是time对象，先转换为字符串，再转换为datetime
                    current_date = datetime.date.today()
                    
                    def convert_time(x):
                        if pd.isna(x):
                            return None
                        try:
                            # 如果是time对象，转换为datetime
                            if isinstance(x, datetime.time):
                                return datetime.datetime.combine(current_date, x)
                            # 如果已经是datetime，直接返回
                            elif isinstance(x, datetime.datetime):
                                return x
                            # 其他情况返回None
                            return None
                        except:
                            return None
                    
                    data['Time'] = data['Time'].apply(convert_time)
                elif isinstance(first_valid_time, datetime.datetime):
                    # 如果已经是datetime，不需要转换
                    pass
                else:
                    # 其他情况，尝试强制转换
                    data['Time'] = pd.to_datetime(data['Time'], errors='coerce')

                # 设置索引前再次检查空值
                if data['Time'].isnull().any():
                    st.warning("时间转换后仍存在空值，将被删除")
                    data = data.dropna(subset=['Time'])
                
                data.set_index('Time', inplace=True)
                columns.remove('Time')
                st.success("Time列已成功设置为索引")
                
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
                data[columns] = data[columns].interpolate(method='linear')
            st.write("空值处理后的数据预览：", data[columns].head(10))

            # 归一化
            normalization = st.selectbox("选择归一化方法", ["不进行归一化", "最小-最大归一化", "Z-score标准化"])
            if normalization == "最小-最大归一化":
                scaler = MinMaxScaler()
                data[columns] = scaler.fit_transform(data[columns])
            elif normalization == "Z-score标准化":
                scaler = StandardScaler()
                data[columns] = scaler.fit_transform(data[columns])
            st.write("归一化后的数据预览：", data[columns].head(10))

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
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # 使用seaborn绘制热图
                    sns.heatmap(corr_matrix, 
                               annot=True,  # 显示相关系数值
                               fmt='.2f',   # 保留两位小数
                               cmap='coolwarm',  # 使用红蓝色图
                               center=0,    # 将0设为中心值
                               square=True, # 保持方形
                               ax=ax)
                    
                    plt.title(f'{corr_method} Correlation Matrix')
                    
                    # 调整布局
                    plt.tight_layout()
                    
                    # 显示图形
                    st.pyplot(fig)
                    plt.close()
                    
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
                    st.write(corr_df)
                    
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
        spectrum_columns = st.multiselect("选择要进行频谱分析的列", data.columns.tolist(), key='spectrum_cols')
        
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
                        
                        signal = data[col].values
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
                        signal = data[col].values
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
                            surf = ax.plot_surface(time_grid, scale_grid, np.abs(coefficients), 
                                                cmap='viridis')
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
            data.columns.tolist()
        )
        
        if density_columns:
            # 选择筛选条件的列（可多选）
            filter_columns = st.multiselect(
                "选择用于筛选数据范围的列（可多选）",
                [col for col in data.columns if col not in density_columns]  # 排除已选择的分析列
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
                        step=(max_val - min_val) / 100
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
                
                # 为每个选择的列创建概率密度分布图
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # 使用不同颜色绘制每个列的分布
                for col in density_columns:
                    # 核密度估计
                    filtered_data[col].plot.kde(ax=ax, linewidth=2, label=f'{col} (KDE)')
                    # 直方图
                    filtered_data[col].hist(ax=ax, density=True, alpha=0.1, bins=30, 
                                         label=f'{col} (Hist)')
                
                # 设置标题和标签
                ax.set_title('Probability Density Distribution')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                
                # 添加网格
                ax.grid(True, alpha=0.3)
                
                # 添加图例（上方）
                ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
                
                # 使用轴坐标添加筛选条件文本（左下角）
                if filter_conditions:
                    filter_text = "筛选条件:\n"
                    for col in filter_columns:
                        range_min, range_max = filter_ranges[col]
                        filter_text += f"{col}: [{range_min:.2f}, {range_max:.2f}]\n"
                    
                    x_min, x_max = ax.get_xlim()
                    y_min, y_max = ax.get_ylim()
                    text_x = x_min + (x_max - x_min) * 0.02
                    text_y = y_min + (y_max - y_min) * 0.02
                    
                    ax.text(text_x, text_y, filter_text,
                           fontsize=8,
                           va='bottom',
                           ha='left',
                           bbox=dict(facecolor='white',
                                   alpha=0.8,
                                   edgecolor='none',
                                   pad=1.5))
                
                # 显示统计信息（右下角）
                stats_text = "统计信息:\n"
                for col in density_columns:
                    stats = filtered_data[col]
                    stats_text += f"\n{col}:\n"
                    stats_text += f"数据量: {len(stats)}\n"
                    stats_text += f"均值: {stats.mean():.2f}\n"
                    stats_text += f"标准差: {stats.std():.2f}\n"
                    stats_text += f"最小值: {stats.min():.2f}\n"
                    stats_text += f"最大值: {stats.max():.2f}\n"
                
                # 获取轴的范围
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                
                # 在右下角添加统计信息
                ax.text(x_max - (x_max - x_min) * 0.02,  # 距离右边界2%
                       y_min + (y_max - y_min) * 0.02,   # 距离下边界2%
                       stats_text,
                       fontsize=10,
                       va='bottom',
                       ha='right',  # 右对齐
                       bbox=dict(facecolor='white',
                               alpha=0.8,
                               edgecolor='none',
                               pad=1.5))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # 显示描述性统计信息
                st.write("描述性统计：")
                desc_stats = filtered_data.describe()
                
                # 重命名索引为中文
                stats_names = {
                    'count': '数量',
                    'mean': '均值',
                    'std': '标准差',
                    'min': '最小值',
                    '25%': '25%分位数',
                    '50%': '中位数',
                    '75%': '75%分位数',
                    'max': '最大值'
                }
                desc_stats.index = [stats_names.get(i, i) for i in desc_stats.index]
                
                st.write(desc_stats)

        # 模型训练部分
        st.subheader("模型训练")
        
        # 添加空选项作为默认值
        target_options = [""] + data.columns.tolist()
        target_column = st.selectbox("选择目标列（预测目标）：", target_options)
        
        # 只有当选择了目标列时才显示后续选项
        if target_column:
            feature_columns = st.multiselect("选择特征列：", [col for col in data.columns if col != target_column])

            model_choice = st.selectbox("选择模型", ["线性回归", "多项式回归", "ARIMA", "梯度下降"])

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

            if st.button("开始训练模型"):
                X = data[feature_columns]
                y = data[target_column]

                # 保持索引的切分
                train_size = int(len(data) * 0.8)
                train_index = data.index[:train_size]
                test_index = data.index[train_size:]
                
                # 按索引切分数据
                X_train = X.loc[train_index]
                X_test = X.loc[test_index]
                y_train = y.loc[train_index]
                y_test = y.loc[test_index]

                if model_choice == "线性回归":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_test_pred)
                    st.write(f"线性回归均方误差: {mse}")

                elif model_choice == "多项式回归":
                    poly = PolynomialFeatures(degree=degree)
                    X_poly_train = poly.fit_transform(X_train)
                    X_poly_test = poly.transform(X_test)
                    model = LinearRegression()
                    model.fit(X_poly_train, y_train)
                    y_train_pred = model.predict(X_poly_train)
                    y_test_pred = model.predict(X_poly_test)
                    mse = mean_squared_error(y_test, y_test_pred)
                    st.write(f"多项式回归（{degree}阶）均方误差: {mse}")

                elif model_choice == "ARIMA":
                    model = sm.tsa.ARIMA(y_train, order=(p, d, q))
                    model_fit = model.fit()
                    y_train_pred = model_fit.predict(start=y_train.index[0], end=y_train.index[-1])
                    y_test_pred = model_fit.forecast(steps=len(y_test))
                    mse = mean_squared_error(y_test, y_test_pred)
                    st.write(f"ARIMA 模型均方误差: {mse}")

                elif model_choice == "梯度下降":
                    model = SGDRegressor(learning_rate='constant', eta0=learning_rate, 
                                       max_iter=max_iter, tol=tol, random_state=random_state)
                    model.fit(X_train, y_train)
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_test_pred)
                    st.write(f"梯度下降模型均方误差: {mse}")

                # 绘制预测值与实际值对比图
                st.subheader("预测结果与实际值对比")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Training set
                ax1.plot(train_index, y_train.values, label='Actual', linewidth=1)
                ax1.plot(train_index, y_train_pred, '--', label='Predicted', linewidth=1)
                ax1.set_title('Training Data')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Value')
                ax1.legend()
                ax1.grid(True)
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
                
                # Test set
                ax2.plot(test_index, y_test.values, label='Actual', linewidth=1)
                ax2.plot(test_index, y_test_pred, '--', label='Predicted', linewidth=1)
                ax2.set_title('Test Data')
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Value')
                ax2.legend()
                ax2.grid(True)
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# 添加侧边栏信息
        
# 添加分隔线
st.sidebar.markdown("---")
        
# 添加开发者信息
st.sidebar.markdown("""
### 开发者信息
        
**开发者**：王康业  
**邮箱**：kangy_wang@hnair.com
        
---
        
### 版权声明
                                     
本应用程序受著作权法和其他知识产权法保护。  
未经授权，禁止复制、修改或分发本程序的任何部分。
                    
Version 1.0.0
""")
        
# 添加一些空行来确保版权信息在底部
st.sidebar.markdown("<br>" * 5, unsafe_allow_html=True)
