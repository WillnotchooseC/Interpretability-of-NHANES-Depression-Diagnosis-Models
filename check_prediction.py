import json
import re

data = json.load(open('./nhanes_model/exp1/test_predictions.json'))

# 1. 统计生成中包含乱码/重复对话的比例
bad = 0
for d in data:
    pred = d['prediction']
    # 找最后一个 # Type: 的位置，看后面有没有乱码
    idx = pred.rfind('# Type:')
    if idx >= 0:
        after = pred[idx+20:]  # 跳过 "# Type: searching" 
        if 'user' in after or '�' in after or 'elementGuid' in after:
            bad += 1

print(f"Contains garbage after valid output: {bad}/100")

# 2. truncate后的平均步数
trunc_steps = []
for d in data:
    pred = d['prediction']
    lines = pred.split('\n')
    last = -1
    for i, l in enumerate(lines):
        if '# Type:' in l:
            last = i
    if last >= 0:
        clean = '\n'.join(lines[:last+1])
        steps = len(re.findall(r'Step \d+:', clean))
        trunc_steps.append(steps)
    else:
        trunc_steps.append(0)

print(f"Avg steps after truncation: {sum(trunc_steps)/len(trunc_steps):.2f}")

# 3. 看几条具体例子
print("\n=== Sample predictions (first 3, truncated view) ===")
for d in data[:3]:
    pred = d['prediction']
    lines = pred.split('\n')
    last = -1
    for i, l in enumerate(lines):
        if '# Type:' in l:
            last = i
    if last >= 0:
        clean = '\n'.join(lines[:last+1])
    else:
        clean = pred[:300]
    
    ref_steps = len(re.findall(r'Step \d+:', d['reference']))
    pred_steps = len(re.findall(r'Step \d+:', clean))
    print(f"\nRef steps: {ref_steps}, Pred steps: {pred_steps}")
    print(f"Pred: {clean[:400]}")
    print("---")