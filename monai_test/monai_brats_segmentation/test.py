import requests

# 接口地址（根据实际部署修改）
API_URL = "http://localhost:8015/predict"  #

# 待处理的文件路径
file_path = "models/imagesTs/spleen_1.nii.gz"

try:
    # 以二进制形式打开文件并上传
    with open(file_path, "rb") as f:
        files = {"file": ("test.nii.gz", f, "application/octet-stream")}
        response = requests.post(API_URL, files=files)

    # 检查响应状态
    if response.status_code == 200:
        result = response.json()
        print("推理成功！返回结果格式：")
        print(f"形状: {result['result_shape']}")
        print(f"数据类型: {result['result_dtype']}")

        # 将返回的列表转回 numpy 数组（实际使用时根据需求处理）
        import numpy as np

        pred_array = np.array(result["result_data"])
        print("结果数组示例（部分数据）：", pred_array)  # 打印部分数据

    else:
        print(f"请求失败，状态码：{response.status_code}，错误信息：{response.text}")

except Exception as e:
    print(f"调用接口时出错：{str(e)}")