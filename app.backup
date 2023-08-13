import pinecone

# init connection to pinecone
pinecone.init(
    api_key="0898750a-ee05-44f1-ac8a-98c5fef92f4a",  # app.pinecone.io
    environment="asia-southeast1-gcp-free"  # find next to api key
)

index_name = "hybrid-image-search"
index = pinecone.GRPCIndex(index_name)

from datasets import load_dataset

# load the dataset from huggingface datasets hub
fashion = load_dataset(
    "ashraq/fashion-product-images-small",
    split='train[:1000]'
)

images = fashion["image"]
metadata = fashion.remove_columns("image")

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
bm25.fit(metadata['productDisplayName'])

from sentence_transformers import SentenceTransformer
import transformers.models.clip.image_processing_clip
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load a CLIP model from huggingface
model = SentenceTransformer(
    'sentence-transformers/clip-ViT-B-32',
    device=device
)



def hybrid_scale(dense, sparse, alpha: float):
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



counter = {"dir_num": 1}
img_files = {'x':[]}

def img_to_file_list(imgs):
  path = "searches"
  sub_path = './' + path + '/' + 'search' + '_' + str(counter["dir_num"])

  # Check whether the specified path exists or not
  isExist = os.path.exists('.'+'/'+path)
    
  if not isExist:
    print("Directory does not exists")
  # Create a new directory because it does not exist
    os.makedirs('.'+'/'+path, exist_ok = True)
    print("The new directory is created!")

  # Check whether the specified path exists or not
  isExist = os.path.exists(sub_path)
    
  if isExist:
    shutil.rmtree(sub_path)

  os.makedirs(sub_path, exist_ok = True)

  img_files = {'search'+str(counter["dir_num"]):[]}
  i = 0
 
  for img in imgs:
    img.save(sub_path+"/img_" + str(i) + ".png","PNG")
    img_files['search'+str(counter["dir_num"])].append(sub_path + '/' + 'img_'+ str(i) + ".png")
    i+=1

  counter["dir_num"]+=1

  return img_files['search'+str(counter["dir_num"]-1)]


import gradio as gr
from deep_translator import GoogleTranslator

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

    with gr.Row():

        text = gr.Textbox(
            value = "ג'ינס כחול לגברים",
            label="Enter the product characteristics:",
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
        #heb_text = GoogleTranslator(source='en', target='iw').translate(eng_text)
        return eng_text
        
    gallery.select( fn=get_select_index, inputs=[text,alpha], outputs=selected )

demo.launch()
