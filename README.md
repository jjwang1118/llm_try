# GPT2 Password Generator

基於 GPT-2 的密碼生成器訓練項目。

## 環境設置

```bash
pip install -r requirements.txt
```

## 下載模型

由於模型文件過大，請手動下載以下模型：

### 預訓練模型

從 Hugging Face 下載模型到 `model/GPT2-Hacker-password-generator/` 目錄：

```bash
# 方法一：使用 git lfs
git lfs install
git clone https://huggingface.co/CodeferSystem/GPT2-Hacker-password-generator model/GPT2-Hacker-password-generator

# 方法二：使用 huggingface_hub
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='CodeferSystem/GPT2-Hacker-password-generator', local_dir='model/GPT2-Hacker-password-generator')"
```

## 下載數據集

由於數據集文件過大，請手動下載 RockYou 數據集：

```bash
# 創建 dataset 目錄
mkdir -p dataset

# 下載 RockYou 數據集（選擇以下方式之一）：
# 1. 從 Kaggle 下載：https://www.kaggle.com/datasets/wjburns/common-password-list-rockyoutxt
# 2. 或從其他來源下載 rockyou.txt 並清理後放入 dataset/ 目錄

# 所需文件：
# - dataset/rockyou-cleaned.txt
# - dataset/rockyou-cleaned-Train.txt
# - dataset/rockyou-cleaned-Train-ready.txt
# - dataset/rockyou-cleaned-Test.txt
```

## 項目結構

```
.
├── data.py                 # 數據處理模組
├── password_vocab.py       # 密碼詞彙處理
├── train.py                # 訓練腳本
├── set.ini                 # 配置文件
├── requirements.txt        # Python 依賴
├── dataset/                # 訓練數據集（需手動下載）
│   ├── rockyou-cleaned.txt
│   ├── rockyou-cleaned-Train.txt
│   ├── rockyou-cleaned-Train-ready.txt
│   └── rockyou-cleaned-Test.txt
├── model/                  # 模型目錄（需手動下載）
│   └── GPT2-Hacker-password-generator/
└── checkpoint/             # 訓練 checkpoints（訓練時自動生成）
```

## 訓練

```bash
python train.py
```

## 參考

- 模型來源：[CodeferSystem/GPT2-Hacker-password-generator](https://huggingface.co/CodeferSystem/GPT2-Hacker-password-generator)
