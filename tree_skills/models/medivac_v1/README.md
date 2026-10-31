# models/medivac_v1

**当前为空——本目录必须只存放由真实教师标签训练并导出的模型。**

放入条件（全部满足）：
1. `TEACHER_MANIFEST.json` 状态 `ready`（真实 MARL-GPT 权重通过结构审计与接口验证）；
2. 真实 A/B 轨迹采集与教师标注完成（DATA_PROTOCOL.md）；
3. `training/train_tree.py` 在批准训练清单上训练，按局验证报告（Top-1 一致率、联合分配一致率、难例）随模型保存；
4. `export/exporter.py` 导出 `model.json`，与原模型数值等价 ≤1e-6（阈值边界/邻近位/极值）；
5. C/D 同观察一致性测试通过。

禁止：随机初始化树、手写规则、合成标签模型冒充训练产物（合成冒烟产物只放 `artifacts/demo/`）。

加载：`tree_skills.medivac.pair_scorer.TreePairScorer("<本目录>")`；schema 版本或特征名单不符将拒绝加载。
