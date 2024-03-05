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

def get_images():

    """
    https://kodlogs.com/34776/json-decoder-jsondecodeerror-expecting-property-name-enclosed-in-double-quotes
    if you try to json.loads() without json.dumps() it will throw an error:
    "Expecting property name enclosed in double quotes"
    """

    google_images = []

    all_script_tags = soup.select("script")

    # # https://regex101.com/r/48UZhY/4
    matched_images_data = "".join(re.findall(r"AF_initDataCallback\(([^<]+)\);", str(all_script_tags)))

    matched_images_data_fix = json.dumps(matched_images_data)
    matched_images_data_json = json.loads(matched_images_data_fix)

    # https://regex101.com/r/VPz7f2/1
    matched_google_image_data = re.findall(r'\"b-GRID_STATE0\"(.*)sideChannel:\s?{}}', matched_images_data_json)

    # https://regex101.com/r/NnRg27/1
    matched_google_images_thumbnails = ", ".join(
        re.findall(r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]',
                   str(matched_google_image_data))).split(", ")

    thumbnails = [
        bytes(bytes(thumbnail, "ascii").decode("unicode-escape"), "ascii").decode("unicode-escape") for thumbnail in matched_google_images_thumbnails
    ]

    # removing previously matched thumbnails for easier full resolution image matches.
    removed_matched_google_images_thumbnails = re.sub(
        r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]', "", str(matched_google_image_data))

    # https://regex101.com/r/fXjfb1/4
    # https://stackoverflow.com/a/19821774/15164646
    matched_google_full_resolution_images = re.findall(r"(?:'|,),\[\"(https:|http.*?)\",\d+,\d+\]", removed_matched_google_images_thumbnails)

    full_res_images = [
        bytes(bytes(img, "ascii").decode("unicode-escape"), "ascii").decode("unicode-escape") for img in matched_google_full_resolution_images
    ]

    return full_res_images


def get_all_image_embeddings_from_urls(dataset, processor, model, device, num_images=100):
    embeddings = []

    # Limit the number of images to process
    dataset = dataset[:num_images]
    working_urls = []

    #for image_url in dataset['image_url']:
    for image_url in dataset:
      if check_valid_URL(image_url):
          try:
              # Download the image
              response = requests.get(image_url)
              image = Image.open(BytesIO(response.content)).convert("RGB")
              # Get the embedding for the image
              embedding = get_single_image_embedding(image, processor, model, device)
              #embedding = get_single_image_embedding(image)
              embeddings.append(embedding)
              working_urls.append(image_url)
          except Exception as e:
              print(f"Error processing image from {image_url}: {e}")
      else:
          print(f"Invalid or inaccessible image URL: {image_url}")

    return embeddings, working_urls

def get_single_image_embedding(image, processor, model, device):
    """
    Get the embedding for a single image
    """
    embedding = processor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.encode_image(embedding)
        output = output / output.norm(dim=-1, keepdim=True)
        embedding = output.cpu().detach()
    return embedding

'''
# 直接将嵌入向量上传到Pinecone索引
def upsert_to_pinecone(index_name, embeddings):
    # 初始化 Pinecone 客户端
    pinecone.init(api_key="your-api-key-here")

    try:
        # 删除旧索引（如果存在）
        pinecone.delete_index(index_name)

        # 创建新的 Pinecone 索引
        pinecone.create_index(index_name=index_name, metric='euclidean')

        # 将所有嵌入插入 Pinecone 索引
        with pinecone.Index(index_name) as index:
            for idx, emb in enumerate(embeddings):
                index.upsert(keys=[str(idx)], values=[emb])

        print(f"Successfully uploaded {len(embeddings)} embeddings to Pinecone index: {index_name}")
    finally:
        pinecone.deinit()
'''

def check_valid_URL(url):
    try:
        response = requests.head(url)
        return response.status_code == 200  # 如果状态码为200，则URL有效
    except requests.exceptions.RequestException:
        return False

# 创建用于在Pinecone中进行更新或插入操作的数据结构
'''
def create_data_to_upsert_from_urls(dataset,  embeddings):
  metadata = []
  image_IDs = []
  for index in range(len(dataset)):
    metadata.append({
        'ID': index,
        'image': dataset[index]
    })
    image_IDs.append(str(index))
  image_embeddings = [arr.tolist() for arr in embeddings]
  data_to_upsert = list(zip(image_IDs, image_embeddings, metadata))
  return data_to_upsert
'''
#试验——改成{id：,values: ,metadata: }进行插入
def create_data_to_upsert_from_urls(dataset, embeddings):
    data_to_upsert = []

    for index in range(len(dataset)):
        metadata = {
            'ID': str(index),
            'image': dataset[index]
        }
        vector = embeddings[index].tolist()[0]  # 确保这里已经是嵌入向量的一维列表

        # 创建一个字典，其中'id'是字符串形式的索引，'values'是嵌入向量，'metadata'是原始元数据字典
        entry = {"id": metadata['ID'], "values": vector, "metadata": metadata}
        data_to_upsert.append(entry)

    return data_to_upsert



def display_images(images_data, ids_scores):
    # 计算相似图片的数量
    n_similar_images = len(images_data) - 1

    # 设置图像显示大小以提高清晰度
    fig, axes = plt.subplots(2, n_similar_images + 1, figsize=(4 * (n_similar_images + 1), 8))
    fig.dpi = 1000
    # 显示输入图像
    input_ax = axes[0, 0]
    input_ax.imshow(images_data[0])
    input_ax.set_title("Input Image", fontsize=3,pad=3)
    input_ax.axis('off')

    # 显示相似图片
    for i in range(n_similar_images):
        similar_ax = axes[1, i]
        similar_ax.imshow(images_data[i + 1],interpolation='bilinear')
        similar_ax.set_title(ids_scores[i + 1], fontsize=1,pad=2)
        similar_ax.axis('off')

    # 隐藏所有轴的刻度和标签
    for ax in axes.flat:
        ax.axis('off')

    # 调整子图间距
    plt.subplots_adjust(hspace=0.5, wspace=0.1)

    # 显示图像
    plt.show()
if __name__ == '__main__':
    # 初始化CLIP模型与预处理器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = clip.load("ViT-B/32", device=device)  # 使用你想要的CLIP模型版本

    """
    # 以下代码用于 网页搜索 获取搜索结果的图片链接
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36"
    }

    params = {
        "q": "the lord of the rings film scenes",  # search query
        "tbm": "isch",  # image results
        "hl": "en",  # language of the search
        "gl": "us",  # country where search comes from
        "ijn": "0"  # page number
    }

    html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=30)
    soup = BeautifulSoup(html.text, "lxml")

    #获取搜索结果中的图片url
    image_urls = get_images()[:100]
    print('get image finished')
    # 获取这些图片的嵌入向量
    embeddings, working_urls = get_all_image_embeddings_from_urls(image_urls, processor, model, device)
    print('get embedding and working_urls finished')
    print(f'len(embeddings)={len(embeddings)}')
    print(f'embeddings[0].shape={(embeddings[0].shape)}')
    # 用api-key初始化 Pinecone 客户端
    pc=Pinecone(api_key="f8f23a71-2acd-4740-81c2-f248855e0724",environment="gcp-starter")
    # 删除旧索引（如果存在），并创建新的 Pinecone 索引
    index_name = 'image-search-ring-of-the-lord'
    vector_dim = embeddings[0].shape[1]
    print(pc.list_indexes())
    '''
    if index_name in pc.list_indexes():
        pc.delete_index(index_name)
        input("删除索引成功，按回车键继续...")
    pc.create_index(name=index_name,dimension=vector_dim,metric="cosine", spec=PodSpec(environment="gcp-starter"))
    '''
    myindex = pc.Index(index_name)#连接到Pinecone索引
    myindex.delete(delete_all=True)
    print('删除索引原数据成功')
    # 创建用于在Pinecone中crud操作的数据结构
    data_to_upsert = create_data_to_upsert_from_urls(working_urls, embeddings)
    myindex.upsert(vectors=data_to_upsert)#插入
    print('插入数据成功')
    try:
        myindex.describe_index_stats()
    except Exception as e:
        print(f'Error describing index stats: {e}')
    #输入图片文件路径
    image_path = input("请输入图像文件路径：")
    try:
        input_image = Image.open(image_path).convert("RGB")
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
    i=0

    for result in results['matches']:
        id_value = result['id']
        metadata = result['metadata']
        score = result['score']
        image_url = metadata['image']
        response = requests.get(image_url)
        img_data = BytesIO(response.content)
        s_image = Image.open(img_data) # 调整合适的大小
        images_data.append(s_image)
        ids_scores.append(f"ID #{id_value} - Score: #{score}")
        print(f'图像{i}读入成功')
        i=i+1


    try:
        display_images(images_data, ids_scores)
        print("显示结果成功！")
    except Exception as e:
        print("无法显示图像。", str(e))
