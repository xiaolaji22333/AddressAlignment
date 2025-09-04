from runner.predict import predict
import pymysql
from pymysql import cursors

MYSQL_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'database': 'address',
    'charset': 'utf8mb4'
}
def base_result(text: str) -> dict:
    # 获取模型预测结果
    pred_results = predict([text])
    if not pred_results:
        return {
            "省份": None,
            "城市": None,
            "区县": None,
            "街道": None,
            "详细地址": text  # 如果没有预测结果，整个文本作为详细地址
        }

    # 提取预测结果
    char_tag_list = pred_results[0]

    # 初始化结果字典
    result = {
        "省份": "",
        "城市": "",
        "区县": "",
        "街道": "",
        "详细地址": ""
    }

    # 按顺序处理每个字符和标签
    for char, tag in char_tag_list:
        if tag == 'O':
            # 非实体字符，跳过
            continue

        # 提取后缀
        if '-' in tag:
            _, suffix = tag.split('-', 1)
        else:
            suffix = None

        # 根据后缀添加到相应的结果字段
        if suffix == "prov":
            result["省份"] += char
        elif suffix == "city":
            result["城市"] += char
        elif suffix in ["devzone", "district"]:
            result["区县"] += char
        elif suffix == "town":
            result["街道"] += char
        elif suffix in ["road", "houseno", "community", "building", "room"]:
            # 将所有地址相关的标签都添加到详细地址
            result["详细地址"] += char

    # 找出所有已识别部分（省份、城市、区县、街道）
    identified_parts = [
        result["省份"],
        result["城市"],
        result["区县"],
        result["街道"]
    ]
    identified_text = "".join([part for part in identified_parts if part])

    # 找出已识别部分在原始文本中的位置
    if identified_text and identified_text in text:
        start_pos = text.find(identified_text)
        end_pos = start_pos + len(identified_text)

        # 将已识别部分之后的内容作为详细地址
        if end_pos < len(text):
            result["详细地址"] = text[end_pos:]
        else:
            result["详细地址"] = ""
    else:
        # 如果没有识别出任何主要部分，将整个文本作为详细地址
        result["详细地址"] = text

    # 如果任何字段为空，设置为None
    for key in result:
        if result[key] == "":
            result[key] = None

    # 返回与接口一致的格式
    result= {
        "省份": result["省份"],
        "城市": result["城市"],
        "区县": result["区县"],
        "街道": result["街道"],
        "详细地址": result["详细地址"]
    }
    return result


def address_alignment(text:str)->dict:
    """
    做二次验证
    :param text:
    :return:
    """
    from _2fa import filter_data
    result = base_result(text)
    filtered_data = filter_data(result)
    return filtered_data

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
    for t in text:
        print(address_alignment(t))

