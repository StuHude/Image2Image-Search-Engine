import requests, lxml, re, json, urllib.request
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import clip
import torch
import pinecone
from pinecone import ServerlessSpec, Pinecone, PodSpec
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
import torchvision.transforms as transforms
from main import display_images, get_single_image_embedding

if __name__ == '__main__':
    # 初始化CLIP模型与预处理器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = clip.load("ViT-B/32", device=device)  # 使用你想要的CLIP模型版本
    # 用api-key初始化 Pinecone 客户端
    pc=Pinecone(api_key="f8f23a71-2acd-4740-81c2-f248855e0724",environment="gcp-starter")
    index_name = 'image-search-ring-of-the-lord'
    print(pc.list_indexes())
    myindex = pc.Index(index_name)
    myindex.describe_index_stats()
    # 输入图片文件路径
    image_path = input("请输入图像文件路径：")
    try:
        input_image = Image.open(image_path).convert("RGB")
        print("图像文件已读取。")
    except FileNotFoundError:
        print("找不到指定的图像文件。")
    except Exception as e:
        print("读取图像时出现错误：", str(e))

    query_embedding = get_single_image_embedding(input_image, processor, model, device).tolist()[0]
    results = myindex.query(vector=query_embedding, top_k=3, include_metadata=True)
    print('查询成功')
    print(results)
    # 获取查询结果并准备存放图片的列表
    images_data = []
    ids_scores = []

    # 将input_image添加到images_data和ids_scores中
    input_image_resized = input_image
    images_data.insert(0, input_image_resized)
    ids_scores.insert(0, "Input Image")
    i = 0

    for result in results['matches']:
        id_value = result['id']
        metadata = result['metadata']
        score = result['score']
        image_url = metadata['image']
        response = requests.get(image_url)
        img_data = BytesIO(response.content)
        s_image = Image.open(img_data)  # 调整合适的大小
        images_data.append(s_image)
        ids_scores.append(f"ID #{id_value} - Score: #{score}")
        print(f'图像{i}读入成功')
        i = i + 1

    print(ids_scores)

    try:
        display_images(images_data, ids_scores)
        print("显示结果成功！")
    except Exception as e:
        print("无法显示图像。", str(e))
