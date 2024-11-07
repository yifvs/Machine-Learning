# Machine-Learning

# 机载数据分析平台

## 项目简介
这是一个基于 Streamlit 开发的机载数据分析平台，用于处理和分析航空数据。该平台提供了多种数据分析功能，包括数据预处理、相关性分析、频谱分析、概率密度分布分析以及机器学习模型训练等功能。

## 主要功能
1. **数据导入**
   - 支持多个 CSV 文件上传
   - 可设置表头和跳过行数
   - 自动合并多个文件

2. **数据预处理**
   - 数值类型转换
   - 空值处理
   - 时间序列处理

3. **相关性分析**
   - Pearson/Spearman 相关系数计算
   - 相关性热图可视化
   - 相关性排序

4. **频谱分析**
   - 快速傅里叶变换（FFT）
   - 频谱图可视化
   - 多变量频谱对比

5. **概率密度分布分析**
   - 核密度估计
   - 直方图可视化
   - 多变量分布对比
   - 条件筛选功能

6. **机器学习模型**
   - 线性回归
   - 多项式回归
   - ARIMA 时间序列模型
   - 梯度下降

## 安装说明
1. 克隆项目到本地
2. 安装依赖包：
pip install -r requirements.txt


## 使用说明
1. 启动应用后，通过侧边栏上传 CSV 格式的数据文件
2. 根据需要选择相应的分析功能
3. 按照界面提示进行操作
4. 查看分析结果和可视化图表

## 系统要求
- Python 3.8 或更高版本
- 参见 requirements.txt 文件获取完整依赖列表

## 开发者信息
- 开发者：王康业
- 邮箱：kangy_wang@hnair.com

## 版权声明

本应用程序受著作权法和其他知识产权法保护。  
未经授权，禁止复制、修改或分发本程序的任何部分。

## 版本信息
- 当前版本：1.0.0
- 发布日期：2024年11月
