import json
failed = json.load(open('memory/failed_factors.json', encoding='utf-8'))
results = []
for f in failed['factors']:
    m = f.get('metrics', {})
    results.append({
        'name': f['factor_name'],
        'sharpe': m.get('sharpe', 0),
        'dd': m.get('max_drawdown', -1),
        'annual': m.get('annual_return', 0),
        'reason': f['reason']
    })
results.sort(key=lambda x: x['sharpe'], reverse=True)
print('Top 10 by Sharpe (Train):')
for r in results[:10]:
    print(f"  {r['name']}: Sharpe={r['sharpe']:.2f} MaxDD={r['dd']:.1%} Annual={r['annual']:.1%}")
