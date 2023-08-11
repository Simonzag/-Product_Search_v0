import pinecone

# init connection to pinecone
pinecone.init(
    api_key="0898750a-ee05-44f1-ac8a-98c5fef92f4a",  # app.pinecone.io
    environment="asia-southeast1-gcp-free"  # find next to api key
)

# index_name = "hybrid-image-search"

# if index_name not in pinecone.list_indexes():
#     # create the index
#     pinecone.create_index(
#       index_name,
#       dimension=512,
#       metric="dotproduct",
#       pod_type="s1"
#     )
index_name = pinecone.list_indexes()[0]
print(index_name)

index = pinecone.GRPCIndex(index_name)

from datasets import load_dataset

# load the dataset from huggingface datasets hub
fashion = load_dataset(
    "ashraq/fashion-product-images-small",
    split='train[:1000]'
)

images = fashion["image"]
metadata = fashion.remove_columns("image")
images[900]

import pandas as pd

metadata = metadata.to_pandas()
filtered = metadata[ (metadata['gender'] == 'Men') & (metadata['articleType'] == 'Jeans')& (metadata['baseColour'] == 'Blue')]
print(len(filtered))
metadata.head()

import requests

with open('pinecone_text.py' ,'w') as fb:
    fb.write(requests.get('https://storage.googleapis.com/gareth-pinecone-datasets/pinecone_text.py').text)

from transformers import BertTokenizerFast
import pinecone_text

# load bert tokenizer from huggingface
tokenizer = BertTokenizerFast.from_pretrained(
    'bert-base-uncased'
)

def tokenize_func(text):
    token_ids = tokenizer(
        text,
        add_special_tokens=False
    )['input_ids']
    return tokenizer.convert_ids_to_tokens(token_ids)

bm25 = pinecone_text.BM25(tokenize_func)

tokenize_func('Turtle Check Men Navy Blue Shirt')

bm25.fit(metadata['productDisplayName'])

display(metadata['productDisplayName'][0])
bm25.transform_query(metadata['productDisplayName'][0])

from sentence_transformers import SentenceTransformer
import transformers.models.clip.image_processing_clip
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load a CLIP model from huggingface
model = SentenceTransformer(
    'sentence-transformers/clip-ViT-B-32',
    device=device
)
model

dense_vec = model.encode([metadata['productDisplayName'][0]])
dense_vec.shape

#len(fashion)

"""##Encode the dataset to index


"""

# from tqdm.auto import tqdm

# batch_size = 200

# for i in tqdm(range(0, len(fashion), batch_size)):
#     # find end of batch
#     i_end = min(i+batch_size, len(fashion))
#     # extract metadata batch
#     meta_batch = metadata.iloc[i:i_end]
#     meta_dict = meta_batch.to_dict(orient="records")
#     # concatinate all metadata field except for id and year to form a single string
#     meta_batch = [" ".join(x) for x in meta_batch.loc[:, ~meta_batch.columns.isin(['id', 'year'])].values.tolist()]
#     # extract image batch
#     img_batch = images[i:i_end]
#     # create sparse BM25 vectors
#     sparse_embeds = [bm25.transform_doc(text) for text in meta_batch]
#     # create dense vectors
#     dense_embeds = model.encode(img_batch).tolist()
#     # create unique IDs
#     ids = [str(x) for x in range(i, i_end)]

#     upserts = []
#     # loop through the data and create dictionaries for uploading documents to pinecone index
#     for _id, sparse, dense, meta in zip(ids, sparse_embeds, dense_embeds, meta_dict):
#         upserts.append({
#             'id': _id,
#             'sparse_values': sparse,
#             'values': dense,
#             'metadata': meta
#         })
#     # upload the documents to the new hybrid index
#     index.upsert(upserts)

# show index description after uploading the documents
index.describe_index_stats()

from IPython.core.display import HTML
from io import BytesIO
from base64 import b64encode
import pinecone_text

# function to display product images
def display_result(image_batch):
    figures = []
    for img in image_batch:
        b = BytesIO()
        img.save(b, format='png')
        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="data:image/png;base64,{b64encode(b.getvalue()).decode('utf-8')}" style="width: 90px; height: 120px" >
            </figure>
        ''')
    return HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    ''')

def hybrid_scale(dense, sparse, alpha: float):
    """Hybrid vector scaling using a convex combination

    alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: a dict of `indices` and `values`
        alpha: float between 0 and 1 where 0 == sparse only
               and 1 == dense only
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def text_to_image(query, alpha, k_results):

  sparse = bm25.transform_query(query)
  dense = model.encode(query).tolist()

  # scale sparse and dense vectors
  hdense, hsparse = hybrid_scale(dense, sparse, alpha=alpha)

  # search
  result = index.query(
      top_k=k_results,
      vector=hdense,
      sparse_vector=hsparse,
      include_metadata=True
  )
  # used returned product ids to get images
  imgs = [images[int(r["id"])] for r in result["matches"]]

  description = []
  for x in result["matches"]:
    description.append( x["metadata"]['productDisplayName'] )

  return imgs, description

def show_dir_content():
  for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import shutil
from PIL import Image
import os

counter = {"dir_num": 1}
img_files = {'x':[]}

def img_to_file_list(imgs):

  os.chdir('/content')

  path = "searches"
  sub_path = 'content/' + path + '/' + 'search' + '_' + str(counter["dir_num"])

  # Check whether the specified path exists or not
  isExist = os.path.exists('content'+'/'+path)
  if not isExist:
    print("Directory does not exists")
  # Create a new directory because it does not exist
    os.makedirs('content'+'/'+path, exist_ok = True)
    print("The new directory is created!")

  #else:
  #  os.chdir('/content/'+path)

  print("Subdir ->The Current working directory is: {0}".format(os.getcwd()))

  # Check whether the specified path exists or not
  isExist = os.path.exists(sub_path)
  if isExist:
    shutil.rmtree(sub_path)

  os.makedirs(sub_path, exist_ok = True)

  img_files = {'search'+str(counter["dir_num"]):[]}
  i = 0
  curr_dir = os.getcwd()
  for img in imgs:
    img.save(sub_path+"/img_" + str(i) + ".png","PNG")
    img_files['search'+str(counter["dir_num"])].append(sub_path + '/' + 'img_'+ str(i) + ".png")

    i+=1

  counter["dir_num"]+=1

  return img_files['search'+str(counter["dir_num"]-1)]

#print(os.getcwd())
# os.chdir('/content/searches')
# print("The Current working directory is: {0}".format(os.getcwd()))
# show_dir_content()

# imgs2, descr = text_to_image('blue jeans for women', 0.5, 4)

# print("The Current working directory is: {0}".format(os.getcwd()))
# show_dir_content()

# img_files = img_to_file_list(imgs2)

# display(img_files)

# print("The Current working directory is: {0}".format(os.getcwd()))
# show_dir_content()

# shutil.rmtree('/content/searches')

# #shutil.rmtree('./content/searches')
# #print("The Current working directory is: {0}".format(os.getcwd()))
# #show_dir_content()
#     #counter, img_files = img_to_file_list(imgs1, counter, img_files)
#     #display(img_files)

# #counter, img_files = img_to_file_list(imgs2)

import gradio as gr
from deep_translator import GoogleTranslator

css = '''
.gallery img {
    width: 45px;
    height: 60px;
    object-fit: contain;
}
'''

counter = {"dir_num": 1}
img_files = {'x':[]}

def fake_gan(text, alpha):
    text_eng=GoogleTranslator(source='iw', target='en').translate(text)
    imgs, descr = text_to_image(text_eng, alpha, 3)
    img_files = img_to_file_list(imgs)
    return img_files

def fake_text(text, alpha):
    en_text = GoogleTranslator(source='iw', target='en').translate(text)
    img , descr = text_to_image(en_text, alpha, 3)
    return descr

with gr.Blocks() as demo:

    with gr.Row():#variant="compact"):

        text = gr.Textbox(
            value = "ג'ינס כחול לגברים",
            label="Enter the product characteristics:",
            #show_label=True,
            #max_lines=1,
            #placeholder="Enter your prompt",
        )

        alpha = gr.Slider(0, 1, step=0.01, label='Choose alpha:', value = 0.05)

    with gr.Row():
        btn = gr.Button("Generate image")

    with gr.Row():
        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(columns=[8], rows=[2], object_fit='scale-down', height='auto')

    with gr.Row():
        selected = gr.Textbox(label="Product description: ", interactive=False, value = "-----> Description <-------",placeholder="Selected")

    btn.click(fake_gan, inputs=[text, alpha], outputs=gallery)

    def get_select_index(evt: gr.SelectData,text,alpha):
        print(evt.index)
        eng_text = fake_text(text, alpha)[evt.index]
        heb_text = GoogleTranslator(source='en', target='iw').translate(eng_text)
        return heb_text

    #gallery.select( get_select_index, None, selected )
    gallery.select( fn=get_select_index, inputs=[text,alpha], outputs=selected )

demo.launch()
#shutil.rmtree('/content/searches')
