# Image2Image-Search-Engine
对于一张图片，在整个互联网上找到和它最相似的n张图片
## 原理
**MLM+爬虫+pinecone**  

**构建以图搜图引擎**

以Unsplash大数据集为基座，再对于上传的图片生成图像描述，进行搜索和爬虫爬取图片，一起作为图片数据集  
通过多模态模型提取嵌入embedding向量，存储进pinecone向量数据库   
最后对于指定图片进行搜索得到embedding相似度前n的图片

## Setting
```
conda create --name yourname
conda activate yourname
pip install -r requirements.txt#配好环境
```

将代码中的pc=Pinecone(api_key=)填上自己的pinecone api_key

```python main.py #启动```  

```python search_only.py#不动数据库，对建好的数据库直接进行以图搜图```


## 目前demo
爬取谷歌搜索"the lord of the rings film scenes"的100张图片  

通过clip模型提取embedding，存入pinecone构建数据集  

对于上传图片进行embedding搜索得到相似度前3的图像

## Todo-list
- [ ]添加Image Captioning，将搜索query换为描述（Blip等生成模型）
- [ ]将Unsplash等大数据集作为基座存入pincone
- [ ]换掉clip模型（demo用clip是因为可以导包直接调用，本地部署换模型，侧重以图搜图应该用 大Image Encoder对小Text Encoder的模型？）
- [ ]索引查询优化（泰森多边形、预先分类...工程问题）
