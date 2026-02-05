import transformers
from datasets import load_dataset



def load_data(dataset_name):
    """
    Load a dataset from the specified directory and dataset name.

    Args:
        data_dir (str): The directory where the dataset is located.
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load (e.g., 'train', 'test').
    Returns:
        Dataset: The loaded dataset split.

    """
    dataset=load_dataset(
        "text",
        data_files={
            "train": f"./dataset/{dataset_name}-cleaned-Train.txt",
            "test": f"./dataset/{dataset_name}-cleaned-Test.txt"
        }
    )

    #格式
    # dataset = {
    #     'train': Dataset({
    #         'text': ['sample text 1', 'sample text 2', ...]
    #     }),
    #     'test': Dataset({
    #         'text': ['sample text 1', 'sample text 2', ...]
    #     })
    # }

    return dataset

def train_eval_dataset(dataset):
    """
    將訓練集拆分為訓練集和驗證集
    
    Args:
        dataset: 原始訓練資料集
    Returns:
        train_ds: 95% 訓練集
        val_ds: 5% 驗證集
    """
    train_and_val = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds = train_and_val["train"]   # 95%
    val_ds = train_and_val["test"]      # 5%
    
    return train_ds, val_ds