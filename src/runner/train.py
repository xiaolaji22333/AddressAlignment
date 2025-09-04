import time
from dataclasses import dataclass
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, \
    AutoModelForTokenClassification
from configuration import config
from preprocess.dataset import get_dataset


@dataclass
class TrainingConfig:
    epochs: int = 10
    output_dir: str = './checkpoint'
    log_dir: str = './logs'
    batch_size: int = 16
    learning_rate: float = 5e-6
    early_stop_metric: str = 'loss'
    use_amp: bool = True
    save_steps: int = 500
    early_stop_patience: int = 5


class Trainer:
    def __init__(self, model,
                 valid_dataset,
                 collate_fn,
                 compute_metrics,
                 device,
                 train_dataset=None,
                 training_config=TrainingConfig()):
        # 训练参数
        self.training_config = training_config
        # 模型和设备
        self.model = model.to(device)
        self.device = device

        # 优化器
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.training_config.learning_rate)

        # 数据集和数据整理器
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.collate_fn = collate_fn

        # 评估函数
        self.compute_metrics = compute_metrics

        # 全局的step数
        self.step = 1

        # Tensorboard
        self.writer = SummaryWriter(
            log_dir=str(Path(self.training_config.log_dir) / time.strftime('%Y-%m-%d-%H-%M-%S')))

        # early_stop
        self.early_stop_best_score = -float('inf')
        self.early_stop_counter = 0

        # amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.training_config.use_amp)

    def _get_dataloader(self, dataset):
        dataset.set_format(type='torch')
        generator = torch.Generator()
        generator.manual_seed(42)
        dataloader = DataLoader(dataset,
                                batch_size=self.training_config.batch_size,
                                shuffle=True,
                                generator=generator,
                                collate_fn=self.collate_fn)
        return dataloader

    def train(self):
        dataloader = self._get_dataloader(self.train_dataset)
        for epoch in range(1, 1 + self.training_config.epochs):
            for index, inputs in enumerate(tqdm(dataloader, desc=f'[Epoch: {epoch}]')):
                loss = self._train_one_step(inputs)
                if self.step % 50 == 0:
                    # 记录loss
                    tqdm.write(f'[Epoch: {epoch} | Step: {self.step}] loss: {loss}')
                    self.writer.add_scalar('loss', loss, self.step)

                    # 验证
                    metrics = self.evaluate()
                    # [Evaluate] loss: 0.1 | accuracy: 0.9 | f1: 0.9
                    metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
                    tqdm.write(f'[Evaluate] {metrics_str}')

                    # 早停判断
                    if self._should_stop(metrics):
                        tqdm.write('早停')
                        return

                self.step += 1

    def _train_one_step(self, inputs):
        self.model.train()
        device = self.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=self.training_config.use_amp):
            outputs = self.model(**inputs)
            loss = outputs.loss
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

    def evaluate(self) -> dict:  # {loss:1.1,accuracy:0.9,f1:0.89}
        dataloader = self._get_dataloader(self.valid_dataset)
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        for inputs in tqdm(dataloader, desc='[Evaluation]'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            loss = outputs.loss
            # 损失
            total_loss += loss.item()
            # 预测结果
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.tolist())
            # 标签
            labels = inputs['labels']
            all_labels.extend(labels.tolist())

        loss = total_loss / len(dataloader)
        metrics = self.compute_metrics(all_predictions, all_labels)
        return {'loss': loss, **metrics}

    def _should_stop(self, metrics):
        metric = metrics[self.training_config.early_stop_metric]
        score = -metric if self.training_config.early_stop_metric == 'loss' else metric
        if score > self.early_stop_best_score:
            self.early_stop_best_score = score
            self.early_stop_counter = 0
            self.model.save_pretrained(str(Path(self.training_config.output_dir) / 'best'))
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= self.training_config.early_stop_patience:
                return True
            else:
                return False


def train():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese')

    # labels
    with open(config.PROCESSED_DATA_DIR / 'labels.txt', 'r', encoding='utf-8') as f:
        all_labels = f.read().split('\n')
    id2label = {index: label for index, label in enumerate(all_labels)}
    label2id = {label: index for index, label in enumerate(all_labels)}
    # 模型
    model = AutoModelForTokenClassification.from_pretrained(config.PRE_TRAINED_DIR / 'bert-base-chinese',
                                                            num_labels=len(all_labels),
                                                            id2label=id2label,
                                                            label2id=label2id)

    # 数据

    train_dataset = get_dataset('train')
    valid_dataset = get_dataset('valid')
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors='pt')

    def compute_metrics(all_predictions, all_labels):
        true_labels = []
        pred_labels = []
        for labels, preds in zip(all_labels, all_predictions):
            for l, p in zip(labels, preds):
                if l != -100:  # 过滤无效标签
                    true_labels.append(l)
                    pred_labels.append(p)

        # 添加 zero_division=0，抑制警告
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

    training_config = TrainingConfig(output_dir=str(config.CHECKPOINT_DIR),
                                     log_dir=str(config.LOGS_DIR),
                                     use_amp=True)

    trainer = Trainer(model=model,
                      train_dataset=train_dataset,
                      valid_dataset=valid_dataset,
                      collate_fn=collate_fn,
                      compute_metrics=compute_metrics,
                      device=device,
                      training_config=training_config)

    trainer.train()


if __name__ == '__main__':
    train()
