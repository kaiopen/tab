# Travelable Area Boundary Dataset

[中文版](./README_ZH.md)

### File Structures
```
.
|-- boundary              // ground truth files of boundaries
|   |-- <sequence>
|   |   |-- <timestamp>.json
|   |   |-- <timestamp>.json
|   |   |-- ...
|   |-- <sequence>
|   |   |-- <timestamp>.json
|   |   |-- <timestamp>.json
|   |   |-- ...
|   |-- ...
|-- pcd                   // point cloud files
|   |-- <sequence>
|   |   |-- <timestamp>.pcd
|   |   |-- <timestamp>.pcd
|   |   |-- ...
|   |-- <sequence>
|   |   |-- <timestamp>.pcd
|   |   |-- <timestamp>.pcd
|   |   |-- ...
|   |-- ...
|-- test.json             // split file of test set
|-- trainval.json         // split file of training-validation set
|-- init_boundary.py      // Python script to initialize the TAB
|-- README.md
|-- README_EN.md

```

### Python Environments
- tqdm
- NumPy
- [PyTorch](https://pytorch.org)
- [KaiTorch](https://github.com/kaiopen/kaitorch)
- [TABKit](https://github.com/kaiopen/tab_kit)

### Initialization

Download [boundary.zip](https://github.com/kaiopen/tab/releases/download/boundary/boundary.zip), [pcd_0.zip](https://github.com/kaiopen/tab/releases/download/PCD_0/pcd_0.zip), [pcd_1.zip](https://github.com/kaiopen/tab/releases/download/PCD_1/pcd_1.zip) and [pcd_2.zip](https://github.com/kaiopen/tab/releases/download/PCD_2/pcd_2.zip).

Compress `boundary.zip`, `pcd_0.zip`, `pcd_1.zip` and `pcd_2.zip`. Make sure the [file structures](#file-structures) are meeting the requirements. Run the command:
```shell
python init_boundary.py
```

NOTE: `pcd_0.zip`, `pcd_1.zip` and `pcd_2.zip` are encrypted. Email yuanxia@njust.edu.cn or kaiopen@foxmail.com for password.
