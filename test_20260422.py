import pandas as pd
import numpy as np
import cvxpy as cp

def optimize_portfolio(expected_returns, benchmark_weights, risk_model, tracking_error_limit=0.05):
    """
    组合优化器：在跟踪误差限制下最大化超额收益
    """
    n = len(expected_returns)
    w = cp.Variable(n) # 投资组合权重
    
    # 目标函数：最大化 Alpha (预期超额收益)
    alpha = expected_returns @ w
    objective = cp.Maximize(alpha)
    
    # 约束条件
    active_weights = w - benchmark_weights
    constraints = [
        cp.sum(w) == 1,              # 满仓约束
        w >= 0,                      # 禁止做空
        cp.quad_form(active_weights, risk_model) <= tracking_error_limit**2  # 限制跟踪误差 (基于协方差)
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return w.value, prob

def calculate_metrics(optimal_weights, benchmark_weights, expected_returns, risk_model, risk_free_rate=0.03):
    """
    计算投资组合的各项指标
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
    
    # 计算 Herfindahl 指数（权重集中度，0-1，越大越集中）
    herfindahl = np.sum(optimal_weights ** 2)
    
    # 计算信息比率（假设跟踪误差不为0）
    information_ratio = (portfolio_alpha - benchmark_alpha) / tracking_error if tracking_error > 1e-10 else 0
    
    # 计算夏普比率
    sharpe_ratio = (portfolio_alpha - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
    
    return {
        'portfolio_alpha': portfolio_alpha,
        'benchmark_alpha': benchmark_alpha,
        'tracking_error': tracking_error,
        'portfolio_volatility': portfolio_volatility,
        'benchmark_volatility': benchmark_volatility,
        'herfindahl': herfindahl,
        'information_ratio': information_ratio,
        'sharpe_ratio': sharpe_ratio
    }

# 示例流程
if __name__ == "__main__":
    # 1. 模拟数据
    n_assets = 50
    assets = [f"Stock_{i}" for i in range(n_assets)]
    
    # 模拟基准权重 (例如沪深300成分股权重)
    benchmark_weights = np.random.rand(n_assets)
    benchmark_weights /= benchmark_weights.sum()
    
    # 模拟因子得分 (预期超额收益 Alpha)
    expected_returns = np.random.normal(0.01, 0.02, n_assets)
    
    # 模拟风险模型 (协方差矩阵)
    cov_matrix = np.random.randn(n_assets, n_assets)
    risk_model = cov_matrix @ cov_matrix.T  # 使其正定
    
    # 2. 执行优化
    optimal_weights, prob = optimize_portfolio(expected_returns, benchmark_weights, risk_model)
    
    # 3. 计算指标
    metrics = calculate_metrics(optimal_weights, benchmark_weights, expected_returns, risk_model)
    
    # 4. 结果分析
    df_result = pd.DataFrame({
        'Asset': assets,
        'Benchmark_Weight': benchmark_weights,
        'Optimal_Weight': optimal_weights,
        'Active_Weight': optimal_weights - benchmark_weights,
        'Expected_Return': expected_returns
    })
    
    # 按权重变化排序
    df_result['Abs_Active_Weight'] = np.abs(df_result['Active_Weight'])
    df_top_changes = df_result.nlargest(10, 'Abs_Active_Weight')[['Asset', 'Benchmark_Weight', 'Optimal_Weight', 'Active_Weight', 'Expected_Return']]
    
    # ===== 输出结果 =====
    print("\n" + "="*80)
    print("【优化结果摘要】")
    print("="*80)
    print(f"优化状态: {prob.status}")
    print(f"\n【目标函数值】")
    print(f"  预期超额收益 (Alpha):     {metrics['portfolio_alpha']:.6f}")
    print(f"  基准 Alpha:              {metrics['benchmark_alpha']:.6f}")
    
    print(f"\n【风险指标】")
    print(f"  投资组合波动率:          {metrics['portfolio_volatility']:.6f}")
    print(f"  基准波动率:              {metrics['benchmark_volatility']:.6f}")
    print(f"  跟踪误差:                {metrics['tracking_error']:.6f} (限制: 0.05)")
    
    print(f"\n【效率指标】")
    print(f"  信息比率:                {metrics['information_ratio']:.6f}")
    print(f"  夏普比率:                {metrics['sharpe_ratio']:.6f}")
    print(f"  权重集中度 (Herfindahl): {metrics['herfindahl']:.6f}")
    
    print(f"\n【约束验证】")
    print(f"  权重和:                  {optimal_weights.sum():.10f} (应等于1.0)")
    print(f"  最小权重:                {optimal_weights.min():.10f} (应≥0)")
    print(f"  最大权重:                {optimal_weights.max():.10f}")
    
    print(f"\n【前5只股票的权重分配】")
    print(df_result[['Asset', 'Benchmark_Weight', 'Optimal_Weight', 'Active_Weight', 'Expected_Return']].head())
    
    print(f"\n【权重变化最大的前10只股票】")
    print(df_top_changes.to_string(index=False))
    
    print("\n" + "="*80)
    
    
    
    
    
    
    
    
    
    
    