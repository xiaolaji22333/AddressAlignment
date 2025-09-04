from datasets import load_dataset, ClassLabel, Dataset, DatasetDict
from transformers import AutoTokenizer

from configuration import config


# 1. 加载原始数据
def load_address_data(file_path):
    samples = []
    current_tokens, current_labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    samples.append({"tokens": current_tokens, "labels": current_labels})
                    current_tokens, current_labels = [], []
                continue
            token, label = line.split()
            current_tokens.append(token)
            current_labels.append(label)
    return samples


# 加载数据并转换为Dataset
data = load_address_data(config.RAW_DATA_DIR / 'data.txt')
dataset = Dataset.from_list(data)

# 2. 直接拆分为训练集、验证集、测试集
# 先拆分出训练集和临时集
train_val_test = dataset.train_test_split(test_size=0.3, seed=42)
# 从临时集中拆分出验证集（10%总数据）和测试集（20%总数据）
val_test = train_val_test['test'].train_test_split(test_size=2 / 3, seed=42)

# 组合为三个子集的DatasetDict
dataset_dict = DatasetDict({
    'train': train_val_test['train'],  # 70%
    'valid': val_test['train'],  # 10%
    'test': val_test['test']  # 20%
})

# 3. 定义标签列表并映射

labels=config.LABELS
label2id = {label: idx for idx, label in enumerate(labels)}

# 定义转换函数：将字符串标签列表转为整数ID列表
def convert_labels(example):
    # 转换为ID
    example['labels'] = [label2id[label] for label in example['labels']]
    return example

# 应用转换
dataset_dict = dataset_dict.map(convert_labels, batched=False)

# 保存标签列表
with open(config.PROCESSED_DATA_DIR / 'labels.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(labels))

# 4. 加载分词器并处理数据
tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')


# 定义分词与标签对齐函数
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        padding="max_length",
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            else:
                label_ids.append(-100)

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 批量处理数据集
tokenized_dataset = dataset_dict.map(tokenize_and_align_labels,batched=True,batch_size=1000,
                                     remove_columns=["tokens", "labels"])

# 转换为torch张量格式
tokenized_dataset = tokenized_dataset.with_format("torch", columns=["input_ids", "attention_mask", "labels"])
print(tokenized_dataset)

# 保存处理后的数据集和分词器
tokenized_dataset.save_to_disk(config.PROCESSED_DATA_DIR)
tokenizer.save_pretrained(config.PROCESSED_DATA_DIR / "tokenizer")

print(f"训练集：{len(tokenized_dataset['train'])}条")
print(f"验证集：{len(tokenized_dataset['valid'])}条")
print(f"测试集：{len(tokenized_dataset['test'])}条")