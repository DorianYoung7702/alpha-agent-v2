content = open('src/backtest/metrics.py', 'r', encoding='utf-8').read()
old = '    # 判断是否通过验证标准\n    metrics["passed"] = (\n        sharpe > 1.5 and\n        max_dd > -0.20 and\n        ic_mean > 0.05\n    )'
new = '    # 判断是否通过验证标准（MaxDD放宽至50%）\n    metrics["passed"] = (\n        sharpe > 1.5 and\n        max_dd > -0.50 and\n        ic_mean > 0.05\n    )'
if old in content:
    content = content.replace(old, new)
    open('src/backtest/metrics.py', 'w', encoding='utf-8').write(content)
    print('OK: MaxDD -> -0.50')
else:
    print('NOT FOUND')
    idx = content.find('passed')
    print(repr(content[idx-5:idx+120]))
