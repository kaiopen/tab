# 可通行区域边缘数据集

[English](./README_EN.md)

### 文件结构
```
.
|-- boundary              // 边缘真值文件
|   |-- <序列名称>
|   |   |-- <时间戳>.json
|   |   |-- <时间戳>.json
|   |   |-- ……
|   |-- <序列名称>
|   |   |-- <时间戳>.json
|   |   |-- <时间戳>.json
|   |   |-- ……
|   |-- ……
|-- pcd                   // 点云文件
|   |-- <序列名称>
|   |   |-- <时间戳>.pcd
|   |   |-- <时间戳>.pcd
|   |   |-- ……
|   |-- <序列名称>
|   |   |-- <时间戳>.pcd
|   |   |-- <时间戳>.pcd
|   |   |-- ……
|   |-- ……
|-- test.json             // 测试集索引列表
|-- trainval.json         // 训练-验证集索引列表
|-- init_boundary.py      // 数据集初始化脚本
|-- README.md
|-- README_EN.md

```

### Python 环境
- tqdm
- NumPy
- [PyTorch](https://pytorch.org)
- [KaiTorch](https://github.com/kaiopen/kaitorch)
- [TABKit](https://github.com/kaiopen/tab_kit)

### 初始化数据集
下载 [boundary.zip](https://github.com/kaiopen/tab/releases/download/boundary/boundary.zip)  [pcd_0.zip](https://github.com/kaiopen/tab/releases/download/PCD_0/pcd_0.zip) [pcd_1.zip](https://github.com/kaiopen/tab/releases/download/PCD_1/pcd_1.zip) 和 [pcd_2.zip](https://github.com/kaiopen/tab/releases/download/PCD_2/pcd_2.zip)

解压真值文件 `boundary.zip` 以及点云文件 `pcd_0.zip` `pcd_1.zip` 和 `pcd_2.zip` 后，确保[文件结构](#文件结构)一致。运行如下指令完成数据集初始化：
```shell
python init_boundary.py
```

注：点云文件 `pcd_0.zip` `pcd_1.zip` 和 `pcd_2.zip` 为加密文件。获取密码请邮件： yuanxia@njust.edu.cn 或 kaiopen@foxmail.com
