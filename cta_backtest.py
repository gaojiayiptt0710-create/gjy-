import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def calculate_momentum_factor(df, window=60):
    """
    计算动量因子：过去window日收益
    动量因子 = (当前价格 / window日前价格) - 1
    """
    df = df.copy()
    df['momentum'] = df.groupby('ticker')['close'].transform(
        lambda x: (x / x.shift(window)) - 1
    )
    return df

def calculate_volatility(df, window=20):
    """
    计算过去window日的收益波动率（可选，用于风险调整）
    """
    df = df.copy()
    df['daily_return'] = df.groupby('ticker')['close'].pct_change()
    df['volatility'] = df.groupby('ticker')['daily_return'].transform(
        lambda x: x.rolling(window=window).std()
    )
    return df

def generate_delta_weights(df, lambda_param=0.05):
    """
    生成权重调整Δw
    1. 横截面标准化动量因子（Z-score）
    2. 风险调整：因子值 / 波动率
    3. 乘以λ参数
    4. 去均值确保权重调整总和为0
    """
    df = df.copy()

    # 对每个日期横截面标准化
    df['momentum_z'] = df.groupby('date')['momentum'].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )

    # 风险调整：如果有波动率，用它归一化，否则用1
    if 'volatility' in df.columns:
        df['risk_adjusted_factor'] = df['momentum_z'] / df['volatility'].fillna(1)
    else:
        df['risk_adjusted_factor'] = df['momentum_z']

    # 生成Δw = λ * 风险调整因子
    df['delta_w'] = lambda_param * df['risk_adjusted_factor']

    # 去均值：确保Δw的横截面均值为0
    df['delta_w'] = df.groupby('date')['delta_w'].transform(
        lambda x: x - x.mean()
    )

    return df

def calculate_portfolio_weights(df):
    """
    计算新权重 = 指数权重 + Δw
    确保long-only，总权重=1
    """
    df = df.copy()

    # 新权重 = 指数权重 + Δw
    df['new_weight'] = df['index_weight'] + df['delta_w']

    # Long-only：权重不能为负
    df['new_weight'] = df['new_weight'].clip(lower=0)

    # 归一化：确保总权重=1
    df['final_weight'] = df.groupby('date')['new_weight'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x
    )

    return df

def backtest_portfolio(df):
    """
    回测组合净值
    计算每日组合收益和累积净值
    """
    df = df.copy()

    # 计算每日品种收益
    df['daily_return'] = df.groupby('ticker')['close'].pct_change()

    # 计算组合每日收益 = Σ(权重 * 品种收益)
    df['portfolio_return'] = df.groupby('date').apply(
        lambda x: (x['final_weight'] * x['daily_return']).sum()
    ).reset_index(level=0, drop=True)

    # 计算指数收益（Beta部分）
    df['index_return'] = df.groupby('date').apply(
        lambda x: (x['index_weight'] * x['daily_return']).sum()
    ).reset_index(level=0, drop=True)

    # 累积净值
    df['portfolio_nav'] = (1 + df['portfolio_return']).cumprod()
    df['index_nav'] = (1 + df['index_return']).cumprod()

    return df

def calculate_performance_metrics(df):
    """
    计算回测绩效指标
    """
    portfolio_returns = df['portfolio_return'].dropna()
    index_returns = df['index_return'].dropna()

    # 年化收益
    ann_return_port = (1 + portfolio_returns.mean()) ** 252 - 1
    ann_return_index = (1 + index_returns.mean()) ** 252 - 1

    # 年化波动率
    ann_vol_port = portfolio_returns.std() * np.sqrt(252)
    ann_vol_index = index_returns.std() * np.sqrt(252)

    # 夏普比率（假设无风险利率=2%）
    rf = 0.02
    sharpe_port = (ann_return_port - rf) / ann_vol_port
    sharpe_index = (ann_return_index - rf) / ann_vol_index

    # 最大回撤
    def max_drawdown(nav):
        peak = nav.expanding().max()
        drawdown = (nav - peak) / peak
        return drawdown.min()

    max_dd_port = max_drawdown(df['portfolio_nav'])
    max_dd_index = max_drawdown(df['index_nav'])

    # Alpha/Beta分解
    # 使用线性回归：组合收益 = alpha + beta * 指数收益
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        index_returns, portfolio_returns
    )
    alpha = intercept * 252  # 年化Alpha
    beta = slope

    return {
        'portfolio': {
            'annual_return': ann_return_port,
            'annual_volatility': ann_vol_port,
            'sharpe_ratio': sharpe_port,
            'max_drawdown': max_dd_port
        },
        'index': {
            'annual_return': ann_return_index,
            'annual_volatility': ann_vol_index,
            'sharpe_ratio': sharpe_index,
            'max_drawdown': max_dd_index
        },
        'alpha_beta': {
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_value**2
        }
    }

def plot_results(df, metrics):
    """
    绘制净值曲线
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['portfolio_nav'], label='CTA Enhanced Portfolio', linewidth=2)
    plt.plot(df['date'], df['index_nav'], label='Commodity Index (Beta)', linewidth=2, alpha=0.7)
    plt.title('CTA Index Enhanced Strategy - Net Value Curve')
    plt.xlabel('Date')
    plt.ylabel('Net Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def generate_sample_data(n_days=500, n_tickers=10):
    """
    生成模拟数据用于回测
    """
    np.random.seed(42)

    # 生成日期
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')

    # 品种列表
    tickers = [f'COMMODITY_{i}' for i in range(n_tickers)]

    # 生成数据
    data = []
    for ticker in tickers:
        # 模拟价格序列（随机游走 + 趋势）
        price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, n_days)))
        # 指数权重（随机，但保持稳定）
        index_weight = np.random.uniform(0.05, 0.15)

        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'ticker': ticker,
                'close': price[i],
                'index_weight': index_weight
            })

    df = pd.DataFrame(data)
    return df

# 主函数：运行完整回测
if __name__ == "__main__":
    print("生成模拟数据...")
    df = generate_sample_data(n_days=500, n_tickers=10)

    print("计算动量因子...")
    df = calculate_momentum_factor(df, window=60)

    print("计算波动率...")
    df = calculate_volatility(df, window=20)

    print("生成权重调整...")
    df = generate_delta_weights(df, lambda_param=0.05)

    print("计算组合权重...")
    df = calculate_portfolio_weights(df)

    print("执行回测...")
    df = backtest_portfolio(df)

    print("计算绩效指标...")
    metrics = calculate_performance_metrics(df)

    print("\n" + "="*60)
    print("CTA指数增强策略回测结果")
    print("="*60)

    print(f"\n【组合绩效】")
    print(f"年化收益:     {metrics['portfolio']['annual_return']:.4f}")
    print(f"年化波动率:   {metrics['portfolio']['annual_volatility']:.4f}")
    print(f"夏普比率:     {metrics['portfolio']['sharpe_ratio']:.4f}")
    print(f"最大回撤:     {metrics['portfolio']['max_drawdown']:.4f}")

    print(f"\n【指数基准】")
    print(f"年化收益:     {metrics['index']['annual_return']:.4f}")
    print(f"年化波动率:   {metrics['index']['annual_volatility']:.4f}")
    print(f"夏普比率:     {metrics['index']['sharpe_ratio']:.4f}")
    print(f"最大回撤:     {metrics['index']['max_drawdown']:.4f}")

    print(f"\n【Alpha/Beta分解】")
    print(f"年化Alpha:    {metrics['alpha_beta']['alpha']:.4f}")
    print(f"Beta:         {metrics['alpha_beta']['beta']:.4f}")
    print(f"R²:           {metrics['alpha_beta']['r_squared']:.4f}")

    print("\n绘制净值曲线...")
    plot_results(df, metrics)

    print("回测完成！")
