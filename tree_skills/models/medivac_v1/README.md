# 医疗配对部署模型

本目录用于放置真实教师标签训练并导出的模型。源码发布不附带训练产物。

加载入口：`tree_skills.medivac.pair_scorer.TreePairScorer(model_dir)`。导出规则使用 `model.json`；原生 sklearn 格式还需模型文件和相应依赖。特征 schema 必须与运行时一致。

部署前应验证教师输入与标签、按对局划分的留出表现、导出数值一致性及完整策略效果。合成测试模型应放在本地实验目录，与部署模型区分。
