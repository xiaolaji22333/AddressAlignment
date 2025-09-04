import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

from configuration import config


class Predictor:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.id2label = model.config.id2label

    def predict(self, text: str | list):
        is_str = isinstance(text, str)
        if is_str:
            text = [text]

        # 1. 分词并保留偏移量（过滤特殊符号需要）
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True  # 获取Token与原始文本的偏移关系
        )
        offset_mapping = inputs.pop('offset_mapping')  # 形状：[batch_size, seq_length, 2]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. 模型推理
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits  # 形状：[batch_size, seq_length, num_labels]

        # 3. 获取每个Token的预测ID（维度：[batch_size, seq_length]）
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        # 4. 转换为标签名称（过滤特殊符号）
        results = []
        for i in range(len(predictions)):
            sample_preds = predictions[i]  # 当前样本的Token预测ID（形状：[seq_length]）
            sample_offsets = offset_mapping[i]  # 当前样本的偏移量（形状：[seq_length, 2]）
            sample_labels = []
            for pred, (start, end) in zip(sample_preds, sample_offsets):
                # 过滤特殊符号（[CLS]、[SEP]、[PAD]的offset是(0,0)）
                if start == 0 and end == 0:
                    continue
                # 转换为标签名称
                sample_labels.append(self.id2label[pred])
            results.append(sample_labels)

        return results[0] if is_str else results


def predict(text):
    model = AutoModelForTokenClassification.from_pretrained(config.CHECKPOINT_DIR / 'best')
    tokenizer = AutoTokenizer.from_pretrained(config.CHECKPOINT_DIR / 'best')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = Predictor(model, tokenizer, device)


    #获取元组列表预测结果
    pre = []
    for t in text:
        # 获取预测的标签列表
        labels = predictor.predict(t)

        # 确保标签数量与字符数量一致
        if len(labels) != len(t):
            if len(labels) < len(t):
                labels.extend(['O'] * (len(t) - len(labels)))
            # 如果标签数量多于字符数量，截断多余的部分
            else:
                labels = labels[:len(t)]

        # 将字符串和标签一一对应，形成元组列表
        char_tag_list = list(zip(t, labels))
        pre.append(char_tag_list)
    return pre

    # 获取字典列表预测结果
    # results = []
    # for t in text:
    #     labels = predictor.predict(t)
    #     if len(labels) != len(t):
    #         if len(labels) < len(t):
    #             labels.extend(['O'] * (len(t) - len(labels)))
    #         else:
    #             labels = labels[:len(t)]
    #     results.append({
    #         "text": t,
    #         "char_label_pairs": [f"{char}:{label}" for char, label in zip(t, labels)]
    #     })
    #
    # return results


if __name__ == '__main__':
    text = [
        "中国浙江省杭州市余杭区葛墩路27号楼",
        "北京市通州区永乐店镇27号楼",
        "北京市市辖区高地街道27号楼",
        "新疆维吾尔自治区划阿拉尔市金杨镇27号楼",
        "甘肃省南市文县碧口镇27号楼",
        "陕西省渭南市华阴市罗镇27号楼",
        "西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
        "广州市花都区花东镇27号楼",
    ]
    prediction_results=predict(text)
    # # 按行输出每个案例的结果
    # for i, result in enumerate(prediction_results, 1):
    #     print(result['char_label_pairs'])