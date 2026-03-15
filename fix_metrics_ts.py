content = open('src/backtest/metrics.py', 'r', encoding='utf-8').read()
# 找到 passed 判断块并替换
import re
pattern = r'    metrics\["passed"\] = \([^)]+\)'
new_block = '''    # 时序趋势跟踪验证标准：Sharpe>1.5, MaxDD<40%
    metrics["passed"] = (
        sharpe > 1.5 and
        max_dd > -0.40
    )'''
result = re.sub(pattern, new_block, content, flags=re.DOTALL)
if result != content:
    open('src/backtest/metrics.py', 'w', encoding='utf-8').write(result)
    print('OK: updated to Sharpe>1.5 and MaxDD>-0.40')
else:
    print('NO CHANGE')
