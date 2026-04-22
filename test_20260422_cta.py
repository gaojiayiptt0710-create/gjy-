import pandas as pd
import numpy as np
import cvxpy as cp

def optimize_cta_portfolio(expected_returns, benchmark_weights, risk_model, tracking_error_limit=0.10):
    """
    CTA组合优化器：在跟踪误差限制下最大化超额收益
    CTA特点：允许做空、高杠杆、期货合约
    """
    n = len(expected_returns)
    w = cp.Variable(n)  # CTA权重（可以正负，代表多空头寸）

    # 目标函数：最大化 Alpha (预期超额收益)
    alpha = expected_returns @ w
    objective = cp.Maximize(alpha)

    # 约束条件
    active_weights = w - benchmark_weights
    constraints = [
        cp.norm(active_weights, 1) <= 2.0,  # CTA杠杆约束：总头寸不超过2倍杠杆
        cp.quad_form(active_weights, risk_model) <= tracking_error_limit**2  # 跟踪误差约束
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return w.value, prob

def calculate_cta_metrics(optimal_weights, benchmark_weights, expected_returns, risk_model, risk_free_rate=0.02):
    """
    计算CTA组合的各项指标
    """
    # 计算预期收益
    portfolio_alpha = expected_returns @ optimal_weights
    benchmark_alpha = expected_returns @ benchmark_weights

    # 计算跟踪误差
    active_weights = optimal_weights - benchmark_weights
    tracking_error = np.sqrt(active_weights @ risk_model @ active_weights)

    # 计算波动率
    portfolio_volatility = np.sqrt(optimal_weights @ risk_model @ optimal_weights)
    benchmark_volatility = np.sqrt(benchmark_weights @ risk_model @ benchmark_weights)

    # 计算杠杆率（总头寸绝对值和）
    leverage = np.sum(np.abs(optimal_weights))

    # 计算信息比率
    information_ratio = (portfolio_alpha - benchmark_alpha) / tracking_error if tracking_error > 1e-10 else 0

    # 计算夏普比率（CTA通常用无风险利率2%）
    sharpe_ratio = (portfolio_alpha - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0

    return {
        'portfolio_alpha': portfolio_alpha,
        'benchmark_alpha': benchmark_alpha,
        'tracking_error': tracking_error,
        'portfolio_volatility': portfolio_volatility,
        'benchmark_volatility': benchmark_volatility,
        'leverage': leverage,
        'information_ratio': information_ratio,
        'sharpe_ratio': sharpe_ratio
    }

# 示例流程
if __name__ == "__main__":
    # 1. 模拟CTA数据
    n_contracts = 20  # CTA交易的期货合约数量
    contracts = [
        'WTI原油', '布伦特原油', '黄金', '白银', '铜', '铝', '玉米', '小麦', '大豆',
        '标普500期货', '纳斯达克期货', '日经225期货', '德国DAX期货', '英镑兑美元',
        '欧元兑美元', '日元兑美元', '澳元兑美元', '加元兑美元', '比特币期货', '以太坊期货'
    ]

    # 模拟基准权重（CTA指数的合约权重）
    benchmark_weights = np.random.rand(n_contracts)
    benchmark_weights /= benchmark_weights.sum()

    # 模拟CTA Alpha信号（基于趋势、均值回归等）
    # CTA预期收益通常更高波动，均值0.5%，标准差3%
    expected_returns = np.random.normal(0.005, 0.03, n_contracts)

    # 模拟期货协方差矩阵（期货波动性更高，相关性复杂）
    cov_matrix = np.random.randn(n_contracts, n_contracts)
    risk_model = cov_matrix @ cov_matrix.T * 0.1  # 放大波动性

    # 2. 执行CTA优化
    optimal_weights, prob = optimize_cta_portfolio(expected_returns, benchmark_weights, risk_model)

    # 3. 计算指标
    metrics = calculate_cta_metrics(optimal_weights, benchmark_weights, expected_returns, risk_model)

    # 4. 结果分析
    df_result = pd.DataFrame({
        'Contract': contracts,
        'Benchmark_Weight': benchmark_weights,
        'Optimal_Weight': optimal_weights,
        'Active_Weight': optimal_weights - benchmark_weights,
        'Expected_Return': expected_returns
    })

    # 按权重变化排序
    df_result['Abs_Active_Weight'] = np.abs(df_result['Active_Weight'])
    df_top_changes = df_result.nlargest(10, 'Abs_Active_Weight')[['Contract', 'Benchmark_Weight', 'Optimal_Weight', 'Active_Weight', 'Expected_Return']]

    # ===== 输出结果 =====
    print("\n" + "="*80)
    print("【CTA组合优化结果摘要】")
    print("="*80)
    print(f"优化状态: {prob.status}")
    print(f"\n【目标函数值】")
    print(f"  预期超额收益 (Alpha):     {metrics['portfolio_alpha']:.6f}")
    print(f"  基准 Alpha:              {metrics['benchmark_alpha']:.6f}")

    print(f"\n【风险指标】")
    print(f"  投资组合波动率:          {metrics['portfolio_volatility']:.6f}")
    print(f"  基准波动率:              {metrics['benchmark_volatility']:.6f}")
    print(f"  跟踪误差:                {metrics['tracking_error']:.6f} (限制: 0.10)")
    print(f"  杠杆率:                  {metrics['leverage']:.6f} (限制: ≤2.0)")

    print(f"\n【效率指标】")
    print(f"  信息比率:                {metrics['information_ratio']:.6f}")
    print(f"  夏普比率:                {metrics['sharpe_ratio']:.6f}")

    print(f"\n【约束验证】")
    print(f"  杠杆约束:                {metrics['leverage']:.6f} (应≤2.0)")

    print(f"\n【前5个合约的权重分配】")
    print(df_result[['Contract', 'Benchmark_Weight', 'Optimal_Weight', 'Active_Weight', 'Expected_Return']].head())

    print(f"\n【权重变化最大的前10个合约】")
    print(df_top_changes.to_string(index=False))

    print("\n" + "="*80)