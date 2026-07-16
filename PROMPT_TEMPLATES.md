# LLM-CFD Prompt 模板

本文件提供 LLM-CFD 中使用的完整 Prompt 模板。所有策略共用同一系统消息，用户消息由 Schema 描述、任务指令和输出格式要求三部分组成，不同策略的差异在于任务指令部分。

## 策略使用说明

| 实验 | 使用的策略 | 说明 |
|------|-----------|------|
| 表7 策略对比 | 零样本 + CoT+少样本 + HSCV | 对比不同策略及HSCV消融 |
| 表15 稳定性运行 | CoT+少样本 | 确定性输出验证（15次独立运行） |
| 表14 HSCV消融 | 少样本 | 对15次少样本运行结果投票聚合 |
| 表17 温度实验 | 少样本 | temp=0.0/0.3/0.5/0.7 各5次 |
| §4.5.5 跨LLM验证 | CoT+少样本 | DeepSeek vs GLM，temp=0.0 |

少样本与 CoT+少样本 在最佳 F1 上无显著差异（均为0.711，初步测试），但 CoT+少样本在 temp=0.0 下 15/15 次运行均生成一致性约束规则（中位数 AUPRC=0.495, IQR=0.000），实现确定性输出。纯少样本策略在 temp=0.0 下存在跨次调用波动（AUPRC范围0.168–0.921），HSCV投票可将中位数提升至0.730。CoT+少样本因确定性输出无需HSCV，推荐作为默认配置。

---

## 系统消息（System Message，所有策略共用）

```
你是一位数据质量专家。请严格按照要求输出JSON格式的结果，不要输出任何其他内容。
```

---

## A.1 共用 Schema 与输出格式

所有策略共用以下前缀（动态注入 Schema 统计信息和 8 行样本数据）：

```
你是一位数据质量专家，专精于条件函数依赖（CFD）的发现。请分析以下数据表的Schema和样本数据，推断可能存在的条件函数依赖关系。

## 数据表Schema（含统计信息）：
{schema_text}

## 数据样本（前8行）：
{sample_text}
```

所有策略共用以下输出格式要求：

```
## 输出格式要求：
请严格以JSON数组格式输出所有发现的CFD规则，每条CFD包含以下字段：
- type: "range"|"enum"|"fd"|"logic"|"consistency"
- condition_attributes: 条件属性名列表
- condition_values: 条件属性的取值（dict格式）
- dependent_attribute: 被依赖属性名
- expected_pattern: 期望取值模式（根据类型不同：range→{"min":X,"max":Y}, enum→{"values":[...]}, fd→{"determines":"value"}, logic→{"expression":"..."}, consistency→{"relation":"..."}）
- confidence_estimate: "high"|"medium"|"low"
- natural_language_description: 中文自然语言描述

请输出JSON数组（仅输出JSON，不要有其他文字）：
```json
[
  {...},
  {...}
]
```
```

---

## A.2 Zero-Shot 策略

在 Schema 和样本之后、输出格式之前，插入以下任务指令：

```
## 任务：
请发现数据中可能存在的所有条件函数依赖关系（CFD），包括但不限于：
- range（范围约束）：数值属性在特定条件下有取值区间
- enum（枚举约束）：类别属性在特定条件下取值受限
- fd（函数依赖）：一个或一组属性值确定另一个属性值
- logic（逻辑约束）：属性间存在逻辑推导关系
- consistency（一致性约束）：属性间存在数值一致性关系
```

---

## A.3 Few-Shot 策略

在 Schema 和样本之后、输出格式之前，插入 10 个参考示例（涵盖全部 5 种 CFD 类型）。以下展示其中 3 个代表性示例：

```
## 参考示例：
示例1 (range): 当Contract='Two year'时，MonthlyCharges在[50,120]区间
→ {"type":"range","condition_attributes":["Contract"],"condition_values":{"Contract":"Two year"},"dependent_attribute":"MonthlyCharges","expected_pattern":{"min":50,"max":120},"confidence_estimate":"high"}

示例2 (enum): 当InternetService='No'时，OnlineSecurity必须为'No internet service'
→ {"type":"enum","condition_attributes":["InternetService"],"condition_values":{"InternetService":"No"},"dependent_attribute":"OnlineSecurity","expected_pattern":{"values":["No internet service"]},"confidence_estimate":"high"}

示例3 (consistency): TotalCharges ≈ tenure × MonthlyCharges
→ {"type":"consistency","condition_attributes":["tenure","MonthlyCharges"],"condition_values":{},"dependent_attribute":"TotalCharges","expected_pattern":{"relation":"TotalCharges ≈ tenure * MonthlyCharges"},"confidence_estimate":"high"}
```

完整 10 个示例（覆盖 range×3、enum×5、logic×1、consistency×1）定义在 `experiment.py` 的 `CFD_EXAMPLES_FEWSHOT` 常量中。

---

## A.4 CoT+Few-Shot 策略

在 Schema 和样本之后、参考示例和输出格式之前，插入七步推理指导，引导 LLM 系统性地覆盖全部约束类型：

```
## 推理步骤（目标：至少20条CFD）：

第一步：识别级联服务依赖（enum，约8条）
  InternetService→OnlineSecurity/OnlineBackup/DeviceProtection/TechSupport/StreamingTV/StreamingMovies
  PhoneService→MultipleLines

第二步：合同类型驱动的数值范围（range，约6条）
  Contract=Month-to-month/One year/Two year → MonthlyCharges和tenure的范围

第三步：人口统计属性枚举（enum，约6条）
  SeniorCitizen/gender/Partner/Dependents/PaperlessBilling/PhoneService的合法取值集合

第四步：业务逻辑相关性（logic，约3条）
  tenure与Churn、SeniorCitizen与Churn、支付方式与PaperlessBilling

第五步：数值一致性约束（consistency，约2条）
  TotalCharges ≈ tenure × MonthlyCharges；tenure=0时TotalCharges≈0

第六步：全局范围约束（range，无条件，约3条）
  MonthlyCharges[18,120]、TotalCharges[0,9000]、tenure[0,72]

第七步：支付与渠道属性枚举（enum，约3条）
  PaymentMethod/InternetService/Contract的合法取值
```

上述推理步骤后接 A.3 的参考示例和共用输出格式要求。

---

## HSCV 说明

HSCV（Hybrid Self-Consistency Voting）机制本身不含独立 Prompt，而是对多次 LLM 调用结果进行投票聚合：
- 类别型规则（fd/enum/logic）：需在 ≥50% 的运行中出现（多数投票）
- 数值/结构型规则（range/consistency）：仅需在 ≥1 次运行中出现（并集）

---

## Adult 数据集 Prompt

UCI Adult Census Income 数据集使用专门的 CoT 风格 prompt，定义在 `adult_hscv_experiment.py` 的 `build_adult_prompt()` 函数中。推理步骤针对 Adult 数据集属性设计：

```
## 推理步骤：
第一步：函数依赖（fd类型）
  education → education_num（学历编码对应）
  marital_status → relationship（婚姻状态决定关系）
  sex → relationship（性别影响关系类别）
  education → occupation（学历影响职业）

第二步：范围约束（range类型）
  age全局范围: [17, 90]
  hours_per_week全局范围: [1, 99]
  hours_per_week按occupation分条件的范围
  capital_gain/capital_loss范围: >= 0

第三步：枚举约束（enum类型）
  workclass, race, sex, marital_status的合法取值集合
  native_country的合法取值

第四步：逻辑约束（logic类型）
  education与income的关系（高学历→高收入概率）
  hours_per_week与income的关系
  age与marital_status的关系
```

---

## 补充实验简化版 Prompt

`supplementary_experiments.py` 中使用更简洁的 prompt 构建器，用于列名消融和稳定性实验。Few-Shot 变体仅含 3 个精简示例（range/enum/logic 各 1 个）。
