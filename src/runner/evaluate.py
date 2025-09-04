import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, DataCollatorWithPadding,AutoModelForTokenClassification

from configuration import config
from preprocess.dataset import get_dataset
from runner.train import Trainer


def evaluate():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.CHECKPOINT_DIR / 'best')

    # 模型
    model = AutoModelForTokenClassification.from_pretrained(config.CHECKPOINT_DIR / 'best')

    # 数据
    test_dataset = get_dataset('test')
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors='pt')

    # 评估函数
    def compute_metrics(all_predictions, all_labels):
        true_labels = []
        pred_labels = []
        for labels, preds in zip(all_labels, all_predictions):
            for l, p in zip(labels, preds):
                if l != -100:  # 过滤无效标签
                    true_labels.append(l)
                    pred_labels.append(p)


        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels,
            average='weighted',
            zero_division=0  # 指定除以零时分母的处理方式（设为0）
        )
        accuracy = accuracy_score(true_labels, pred_labels)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    trainer = Trainer(model=model,
                      valid_dataset=test_dataset,
                      collate_fn=collate_fn,
                      compute_metrics=compute_metrics,
                      device=device)

    metrics = trainer.evaluate()
    print(metrics)


if __name__ == '__main__':
    evaluate()
