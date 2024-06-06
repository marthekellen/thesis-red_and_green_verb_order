# install necessary libraries
!pip install transformers
!pip install plotly
!pip install scikit-learn
!pip install nbformat

# import required libraries
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from urllib.request import urlretrieve
import plotly.express as px
from sklearn.decomposition import PCA

# set device to GPU if available
cuda = torch.device('cuda')

# load tokenizer and model from pre-trained BERTje model
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased")

# move model to GPU
model.cuda()

# Disable gradient calculations to save memory, as we are not training the model
# tensor with ids are moved to the GPU
@torch.no_grad()
def encode_sentence_and_extract_position(sentence, position):
    ids = tokenizer.encode(sentence)
    bert_output = model.forward(torch.tensor(ids, device=cuda).unsqueeze(0),
                                encoder_hidden_states = True)
    final_layer_embeddings = bert_output['last_hidden_state'].squeeze()
    return final_layer_embeddings[position].unsqueeze(0)

# Download the dataset from the CCL server
FILENAME = "RoodGroenAnthe.csv"
DATASET_URL = f"https://www.ccl.kuleuven.be/SONAR/{FILENAME}"
urlretrieve(DATASET_URL, FILENAME)

# Read dataset with pandas
df = pd.read_csv(FILENAME)
df.head()

# Function to filter sentences by lemma
def get_sentences_by_lemma(lemma):
  sentences = []
  return df[df["participle_lemma"] == lemma] 

# Example usage to get sentences with the lemma 'schrijven'
schrijven_sentences = get_sentences_by_lemma("schrijven")
schrijven_sentences

# Function to get embeddings from sentences
def get_embeddings_from_sentences(sentences):
  embeddings = []
  ids = []
  for sentence_id, sentence in sentences.iterrows():
    try:
      embeddings.append(encode_sentence_and_extract_position(sentence["sentence"], sentence["participle_index"]))
    # print sentences id if there is an error
    except:
      print(sentence["sentence_id"])
      ids.append(sentence["sentence_id"])
  return embeddings, ids

# Function to create a new datafram with PCA values
def get_df_with_pca(sentences, n_sample=False, random_state=42):
  # sample a subset of the sentences if specified
  if n_sample:
    df = sentences.sample(n_sample, random_state=random_state)
  # otherwise use all sentences
  else:
    df = sentences.copy()

  embeddings, indexes = get_embeddings_from_sentences(df)
  for ids in indexes:
    df = df[df.sentence_id != ids]
    print(ids)
  
  emb_matrix = torch.cat(embeddings, dim=0)
  matrix_np = emb_matrix.cpu().numpy()

  # reduce dimensionality to 3 dimensions
  pca = PCA(n_components=3)
  components = pca.fit_transform(matrix_np)
  #print(pca.explained_variance_ratio_[:15])
  #print(pca.singular_values_[:15])
  
  # insert PCA components in dataframe
  df.insert(1, 'x', components[:,0])
  df.insert(2, 'y', components[:,1])
  df.insert(3, 'z', components[:,2])

  return df

# Example usage: get dataframe for the sentences with 'schrijven'
df_schrijven = get_df_with_pca(schrijven_sentences)

# Count red and green verb orders for 'schrijven'
red_order_count = df_schrijven[df['order'] == 'red'].shape[0]
green_order_count = df_schrijven[df['order'] == 'green'].shape[0]
print(red_order_count)
print(green_order_count)

#  Create an interactive 2D scatter plot (hover over points to display sentences)
fig = px.scatter(df_schrijven, x='x', y='y', color='order',
                 color_discrete_map={'red': 'red', 'green': 'green'},
                 hover_data='sentence')
fig.show()

# Create an interactive 3D scatter plot
fig = px.scatter_3d(
    df_schrijven, x='x', y='y', z='z', color='order',
    color_discrete_map={'red': 'red', 'green': 'green'},
    hover_data='sentence'
)
fig.show()

# Add sentences with switched orders to the dataframe
order = 'switched orders'
new_row = {'sentence': 'Ik moet eerst spreken met de man die dat artikel geschreven heeft.',
              'participle': 'geschreven',
              'auxiliary': 'heeft',
              'participle_lemma': 'schrijven',
              'auxiliary_lemma': 'hebben',
              'participle_index': 11,
              'auxiliary_index': 12,
              'order': order
              }

new_row_1 = {'sentence': 'Als auteur weet je uiteindelijk zelf wel of je een goed of een slecht boek geschreven hebt.',
              'participle': 'geschreven',
              'auxiliary': 'hebt',
              'participle_lemma': 'schrijven',
              'auxiliary_lemma': 'hebben',
              'participle_index': 16,
              'auxiliary_index': 17,
              'order': order
              }

new_row_2 = {'sentence': 'Critici roepen dan meteen dat ik een boek over autisme geschreven heb.',
              'participle': 'geschreven',
              'auxiliary': 'heb',
              'participle_lemma': 'schrijven',
              'auxiliary_lemma': 'hebben',
              'participle_index': 11,
              'auxiliary_index': 12,
              'order': order
              }

new_row_3 = {'sentence': 'Ik vind het mooi, maar als ik het geschreven had, had het veel beter moeten zijn.',
              'participle': 'geschreven',
              'auxiliary': 'had',
              'participle_lemma': 'schrijven',
              'auxiliary_lemma': 'hebben',
              'participle_index': 9,
              'auxiliary_index': 10,
              'order': order
              }

new_row_4 = {'sentence': 'Hoe konden ze te weten komen dat jij me al die brieven geschreven hebt?',
              'participle': 'geschreven',
              'auxiliary': 'hebt',
              'participle_lemma': 'schrijven',
              'auxiliary_lemma': 'hebben',
              'participle_index': 13,
              'auxiliary_index': 14,
              'order': order
              }

# Append the new rows with switched orders
schrijven_sentences_copy = schrijven_sentences.copy()
df_schrijven_switched_sentences = schrijven_sentences_copy._append(new_row, ignore_index=True)
df_schrijven_switched_sentences = df_schrijven_switched_sentences._append(new_row_1, ignore_index=True)
df_schrijven_switched_sentences = df_schrijven_switched_sentences ._append(new_row_2, ignore_index=True)
df_schrijven_switched_sentences = df_schrijven_switched_sentences ._append(new_row_3, ignore_index=True)
df_schrijven_switched_sentences  = df_schrijven_switched_sentences ._append(new_row_4, ignore_index=True)

# Get PCA dataframe for sentences with switched orders
df_schrijven_copyswitched_sentences  = get_df_with_pca(df_schrijven_switched_sentences )
df_schrijven_copyswitched_sentences.head()

# Create an interactive 2D scatter plot for sentences with switched orders
fig = px.scatter(df_schrijven_copyswitched_sentences , x='x', y='y', color='order',
                 color_discrete_map={'red': 'red', 'green': 'green', 'switched orders':'purple'},
                 hover_data='sentence')
fig.show()

# Function to map auxiliary verbs to colors for plotting
def custom_color_mapping(auxiliary_lemma):
  if auxiliary_lemma == 'worden':
    return 'worden'
  elif auxiliary_lemma == 'zijn':
    return 'zijn'
  elif auxiliary_lemma == 'hebben':
    return 'hebben'
color_discrete_map = {'hebben': 'rgb(255,0,0)', 'worden': 'rgb(0, 128, 0)', 'other verb': 'rgb(0, 0, 255)'}
color=df_schrijven['auxiliary_lemma'].apply(custom_color_mapping)

# Inspect the distribution of the auxiliary verbs
fig = px.scatter(df_schrijven, x='x', y='y', color=color, color_discrete_map = color_discrete_map, hover_data='sentence')
fig.show()

# Function to map auxiliary verb and order to colors for plotting
def custom_color_mapping(auxiliary_lemma, order):
    if order == 'red':
        if auxiliary_lemma == 'worden':
            return 'worden_red'
        elif auxiliary_lemma == 'zijn':
            return 'zijn_red'
        elif auxiliary_lemma == 'hebben':
            return 'hebben_red'
        else: 
            return 'other verb_red'
    elif order == 'green':
        if auxiliary_lemma == 'worden':
            return 'worden_green'
        elif auxiliary_lemma == 'zijn':
            return 'zijn_green'
        elif auxiliary_lemma == 'hebben':
            return 'hebben_green'
        else: 
            return 'other verb_green'

df_schrijven['color'] = df_schrijven.apply(lambda row: custom_color_mapping(row['auxiliary_lemma'], row['order']), axis=1)

# Create scatter plot using Plotly Express
fig = px.scatter(df_schrijven, x='x', y='y', color='color', color_discrete_map=color_discrete_map, hover_data='sentence')

# Show the plot
fig.show()

# Initialize counters for auxiliary verbs in red and green orders
counter_worden_green = 0
counter_worden_red = 0
counter_hebben_green = 0
counter_hebben_red = 0
counter_zijn_green = 0
counter_zijn_red = 0
counter_other_green = 0
counter_other_red = 0

# Count occurrences of auxiliary verbs in red and green orders
for sentence_id, el in df_schrijven.iterrows():
    if el['auxiliary_lemma'] == 'worden':
        if el['order'] == 'red':
            counter_worden_red += 1
        elif el['order'] == 'green':
            counter_worden_green +=1
    elif el['auxiliary_lemma'] == 'hebben':
        if el['order'] == 'red':
            counter_hebben_red += 1
        elif el['order'] == 'green':
            counter_hebben_green +=1
    elif el['auxiliary_lemma'] == 'zijn':
        if el['order'] == 'red':
            counter_zijn_red += 1
        elif el['order'] == 'green':
            counter_zijn_green +=1
    else:
        if el['order'] == 'red':
            counter_other_red += 1
        elif el['order'] == 'green':
            counter_other_green +=1

print(counter_worden_green)
print(counter_worden_red)
print(counter_hebben_green)
print(counter_hebben_red)
print(counter_zijn_green)
print(counter_zijn_red)
print(counter_other_green)
print(counter_other_red)

# Function to create a PCA dataframe for a specific verb
def df_from_verb_pca(verb):
  df_sentences = get_sentences_by_lemma(verb)
  return get_df_with_pca(df_sentences)

# Illustration of function: create a PCA dataframe for a specific verb
df_verb = df_from_verb_pca('sluiten')

# Create an interactive 2D scatter plot for the verb 'sluiten'
fig = px.scatter(df_verb, x='x', y='y', color='order',
                 color_discrete_map={'red': 'red', 'green': 'green'},
                 hover_data='sentence')
fig.show()
