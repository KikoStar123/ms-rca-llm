$ErrorActionPreference = "Stop"

$root = "d:\Graduation_Project"
Set-Location $root

Write-Host "== 端到端流程（小样例）=="

Write-Host "1) 解压数据"
python pipeline\extract_day.py --date 2020-04-11

Write-Host "2) 生成 SFT 样本"
python pipeline\build_sft_samples.py --date 2020-04-11 --output output\sft_samples_2020_04_11.jsonl

Write-Host "3) 生成 LLM 输入样本 (v4 Top-K)"
python pipeline\build_llm_inputs.py --input-dir output --pattern "sft_samples_2020_04_11.jsonl" --output output\llm_inputs_v4.jsonl --anomaly-once

Write-Host "4) 输出格式校验"
python pipeline\validate_rca_output.py --input output\llm_inputs_v4.jsonl --output output\rca_output_v4.jsonl --report output\rca_output_v4_report.json

Write-Host "== 完成 =="
