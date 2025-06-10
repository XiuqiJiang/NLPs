import re

input_file = "data/generated_NLPs.txt"
output_file = "data/generated_NLPs_clean.txt"

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        # 匹配"[扰动X] 生成: ... | 编辑距离: ..."格式
        match = re.search(r"生成:\s*([A-Z ]+)\s*\|", line)
        if match:
            seq = match.group(1).replace(" ", "")  # 去掉空格
            fout.write(seq + "\n")

print(f"已提取所有生成序列到 {output_file}")
