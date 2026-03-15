content = open('combined_strategy.py', 'r', encoding='utf-8').read()
print('adx lines:')
for i, line in enumerate(content.split('\n')):
    if 'adx' in line.lower():
        print(f'{i}: {repr(line)}')
