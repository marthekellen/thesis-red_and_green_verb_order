# install necessary libraries
!pip install transformers
!pip install plotly
!pip install scikit-learn
!pip install nbformat

# import required libraries
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
cuda = torch.device('cuda')

# load the tokenizer from BERTje
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased")

model.cuda()

# function to encode a sentence and extract the embeddings at a specific position
def encode_sentence_and_extract_position_cuda(sentence, position):
    ids = tokenizer.encode(sentence)
    bert_output = model.forward(torch.tensor(ids, device=cuda).unsqueeze(0),
                                encoder_hidden_states = True)
    final_layer_embeddings = bert_output['last_hidden_state'].squeeze()
    return final_layer_embeddings[position].unsqueeze(0)
import pandas as pd

# function to get the embeddings for a list of sentences
def get_embeddings_from_sentences(sentences):
  embeddings = []
  for index, sentence in sentences.iterrows():
    embeddings.append(encode_sentence_and_extract_position_cuda(sentence["sentence"], sentence["participle_index"]))

  return embeddings

# function to create a new dataframe with PCA values
def get_df_with_pca(sentences, n_sample=False, random_state=42):
  if n_sample:
    df = sentences.sample(n_sample, random_state=random_state)
  else:
    df = sentences.copy()

  # get the embeddings for the sentences
  embeddings = get_embeddings_from_sentences(df)
  emb_matrix = torch.cat(embeddings, dim=0)
  matrix_np = emb_matrix.cpu().detach().numpy()

  # perform PCA to reduce dimensionality to 3 dimensions
  pca = PCA(n_components=3)
  components = pca.fit_transform(matrix_np)

  df.insert(1, 'x', components[:,0])
  df.insert(2, 'y', components[:,1])
  df.insert(3, 'z', components[:,2])

  return df

# list of example sentences
list_examples = ['Ik hoor dat je mijn boek gelezen hebt.'
, 'Ik hoor dat je mijn boek hebt gelezen.'
,'Ik hoor dat je mijn grote zus geschreven hebt.'
,'Ik hoor dat je mijn grote zus hebt geschreven.'
,'Je kan zien dat het schilderij gerestaureerd is.'
,'Je kan zien dat het schilderij is gerestaureerd.'
,'Ik hoop dat het pakje snel geleverd wordt.'
,'Ik hoop dat het pakje snel wordt geleverd.'
, 'Hij ziet dat het meisje gepest wordt.'
, 'Hij ziet dat het meisje wordt gepest.'
,'Je kan zien dat hij hard voor het examen gestudeerd heeft.'
,'Je kan zien dat hij hard voor het examen heeft gestudeerd.'
,'Hij vertelt dat hij vannacht eng gedroomd heeft. '
,'Hij vertelt dat hij vannacht eng heeft gedroomd.'
,'Ik ruik de geur van het brood dat gebakken wordt.'
,'Ik ruik de geur van het brood dat wordt gebakken.'
,'Hij gelooft niet dat ze zonder studeren geslaagd is.'
,'Hij gelooft niet dat ze zonder studeren is geslaagd.'
,'De politie komt klagen omdat er teveel lawaai gemaakt wordt.'
,'De politie komt klagen omdat er teveel lawaai wordt gemaakt.'
,'Ik weet niet hoe laat ze vertrokken is.'
,'Ik weet niet hoe laat ze is vertrokken.'
,'Hij zegt dat hij naar school gestapt is.'
,'Hij zegt dat hij naar school is gestapt.'
,'Ik zie dat hij veel gedronken heeft.'
,'Ik zie dat hij veel heeft gedronken.'
,'Hij toont dat hij hard gewerkt heeft.'
,'Hij toont dat hij hard heeft gewerkt.'
,'De leraar wist niet waarom de student zo hard geschreeuwd heeft tijdens de les.'
,'De leraar wist niet waarom de student zo hard heeft geschreeuwd tijdens de les.'
,'Het jongetje is verdrietig omdat zijn konijn gestorven is.'
,'Het jongetje is verdrietig omdat zijn konijn is gestorven.'
,'De oude man vertelde dat hij omwille van de losliggende stoeptegel gevallen is.'
,'De oude man vertelde dat hij omwille van de losliggende stoeptegel is gevallen.'
,'Hij gelooft niet dat de dokter hem genezen heeft.'
,'Hij gelooft niet dat de dokter hem heeft genezen.'
,'Hij vindt het heel erg wat haar overkomen is.'
,'Hij vindt het heel erg wat haar is overkomen.'
,'De burgemeester overhandigt hem een medaille aangezien hij zo veel geholpen heeft.'
,'De burgemeester overhandigt hem een medaille aangezien hij zo veel heeft geholpen.'
,'De koning verneemt dat het merendeel van zijn soldaten gesneuveld is.'
,'De koning verneemt dat het merendeel van zijn soldaten is gesneuveld.'
,'De leerkracht geeft toe dat de opmerking hem verbaasd heeft.'
,'De leerkracht geeft toe dat de opmerking hem heeft verbaasd.'
,'Zijn vader vroeg hem of hij de uitnodiging nu eindelijk verstuurd heeft.'
,'Zijn vader vroeg hem of hij de uitnodiging nu eindelijk heeft verstuurd.'
,'De moeder is trots dat de erepenning aan haar zoon overhandigd wordt.'
,'De moeder is trots dat de erepenning aan haar zoon wordt overhandigd.'
,'De student is opgelucht dat het vraagstuk hem eindelijk gelukt is.'
,'De student is opgelucht dat het vraagstuk hem eindelijk is gelukt.'
, 'De directeur vindt dat hij de laatste tijd hard gewerkt heeft.'
, 'De directeur vindt dat hij de laatste tijd hard heeft gewerkt.'
, 'Het kindje vertelt dat ze een hele dag aan de tekening gewerkt heeft.'
, 'Het kindje vertelt dat ze een hele dag aan de tekening heeft gewerkt.'
, 'De student geeft toe dat ze vele uren gewerkt heeft aan het voltooien van haar paper.'
, 'De student geeft toe dat ze vele uren heeft gewerkt aan het voltooien van haar paper.'
, 'Ze aanhoorde het gruwelijke verhaal over hoe hij doodgebloed was op het slagveld.'
, 'Ze aanhoorde het gruwelijke verhaal over hoe hij was doodgebloed op het slagveld.'
, 'De knecht riep door het kasteel dat de prinses ontwaakt was.'
, 'De knecht riep door het kasteel dat de prinses was ontwaakt.'
, 'Het is triest dat de atleet tijdens de laatste horde gestruikeld was.'
, 'Het is triest dat de atleet tijdens de laatste horde was gestruikeld.'
, 'De boer zei dat toen hij de koe kocht hij verhongerd was.'
, 'De boer zei dat toen hij de koe kocht hij was verhongerd'
, 'Het is een feit dat door de storm vele bomen gesneuveld zijn.'
, 'Het is een feit dat door de storm vele bomen zijn gesneuveld.'
, 'Hij hoorde op de radio dat bij het ongeval vele mensen gesneuveld zijn.'
, 'Hij hoorde op de radio dat bij het ongeval vele mensen zijn gesneuveld.'
, 'Ergens is hij opgelucht dat de kat vredig in haar slaap gestorven is.'
, 'Ergens is hij opgelucht dat de kat vredig in haar slaap is gestorven.'
, 'Ze zei dat haar grootouders beide op hoge leeftijd gestorven zijn.'
, 'Ze zei dat haar grootouders beide op hoge leeftijd zijn gestorven.'
, 'Het jongetje is verdrietig omdat zijn konijnen gestorven zijn.'
,'Het jongetje is verdrietig omdat zijn konijnen zijn gestorven.'
, 'Ergens is hij opgelucht dat de katten vredig in hun slaap gestorven zijn.'
, 'Ergens is hij opgelucht dat de katten vredig in hun slaap zijn gestorven.'
, 'Ze zei dat haar oma op hoge leeftijd gestorven is.'
, 'Ze zei dat haar oma op hoge leeftijd is gestorven.'
, 'De koning verneemt dat zijn soldaten gesneuveld zijn.'
, 'De koning verneemt dat zijn soldaten zijn gesneuveld.'
, 'Het is een feit dat de boom door de storm gesneuveld is.'
, 'Het is een feit dat de boom door de storm is gesneuveld.'
, 'Hij hoorde op de radio dat bij het ongeval een persoon gesneuveld is.'
, 'Hij hoorde op de radio dat bij het ongeval een persoon is gesneuveld.'
, 'Tijdens het klasbezoek, vertelde de agent dat hij bevoegd is om het verkeer te regelen.'
, 'Tijdens het klasbezoek, vertelde de agent dat hij is bevoegd om het verkeer te regelen.'
, 'De eigenaar van de boerderij is triest omdat zijn kat vermist is.'
, 'De eigenaar van de boerderij is triest omdat zijn kat is vermist.'
, 'De sollicitant gaf aan dat hij bereid is hard te werken voor de job.'
, 'De sollicitant gaf aan dat hij is bereid hard te werken voor de job.'
, 'De moeder voelde aan dat de leerkracht erg betrokken is met haar leerlingen.'
, 'De moeder voelde aan dat de leerkracht erg is betrokken met haar leerlingen.'
, 'Hij had niet door dat de winkel al een hele tijd gesloten is.'
, 'Hij had niet door dat de winkel al een hele tijd is gesloten.'
, 'Ik geloof dat de winkel gesloten is.'
, 'Ik geloof dat de winkel is gesloten.'
, 'Hij ziet dat de winkel gesloten wordt.'
, 'Hij ziet dat de winkel wordt gesloten.'
, 'Ik zie dat het pakje geleverd is.'
, 'Ik zie dat het pakje is geleverd.'
, 'Je kan zien dat het schilderij gerestaureerd wordt.'
, 'Je kan zien dat het schilderij wordt gerestaureerd.'
                  ]

# list of indices of the participles in the sentences
list_indices = [7,8,7,8,7,8,7,8,6,7,10,11,7,8,9,10,8,9,9,10,7,8,7,8,
                6,7,6,7,10,11,8,9,12,13,8,9,8,9,11,12,10,11,9,10,11,
                12,11,12,10,11,10,11,12,13,9,10,9,10,10,11,11,12,11, 
                12,11,12,12,13,12,13,10,11,9,10,12,13,9,10,7,8,11,12,
                12,13,9,10,11,12,7,8,9,10,12,13,6,7,6,7,6,7,7,8
                       ]

# list of verb orders of the sentences
list_colors = ['green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red',
                'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red',
                'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red',
                'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red',
                'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red',
                'green', 'red', 'green', 'red', 'green', 'red',  'green', 'red', 'green', 'red', 
                'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 
                'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 
                'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red',
                'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red', 'green', 'red',
                'green', 'red'
                ]

# list of past participles of the sentences
list_participles = ['gelezen', 'gelezen', 'geschreven', 'geschreven', 'gerestaureerd', 'gerestaureerd', 'geleverd', 'geleverd', 'gepest', 'gepest',
                     'gestudeerd', 'gestudeerd', 'gedroomd', 'gedroomd', 'gebakken', 'gebakken', 'geslaagd', 'geslaagd', 'gemaakt', 'gemaakt',
                     'vertrokken', 'vertrokken', 'gestapt', 'gestapt', 'gedronken', 'gedronken', 'gewerkt', 'gewerkt', 'geschreeuwd', 'geschreeuwd',
                     'gestorven', 'gestorven', 'gevallen', 'gevallen', 'genezen', 'genezen', 'overkomen', 'overkomen', 'geholpen', 'geholpen',
                     'gesneuveld', 'gesneuveld', 'verbaasd', 'verbaasd', 'verstuurd', 'verstuurd', 'overhandigd', 'overhandigd', 'gelukt', 'gelukt',
                     'gewerkt', 'gewerkt','gewerkt','gewerkt','gewerkt','gewerkt', 'doodgebloed', 'doodgebloed', 'ontwaakt', 'ontwaakt', 'gestruikeld',
                     'gestruikeld', 'verhongerd', 'verhongerd', 'gesneuveld', 'gesneuveld', 'gesneuveld', 'gesneuveld', 'gestorven', 'gestorven',
                     'gestorven', 'gestorven', 'gestorven', 'gestorven', 'gestorven', 'gestorven', 'gestorven', 'gestorven', 'gesneuveld', 'gesneuveld',
                     'gesneuveld', 'gesneuveld', 'gesneuveld', 'gesneuveld', 'bevoegd', 'bevoegd', 'vermist', 'vermist', 'bereid', 'bereid', 'betrokken',
                     'betrokken', 'gesloten', 'gesloten', 'gesloten', 'gesloten', 'gesloten', 'gesloten', 'geleverd', 'geleverd', 'gerestaureerd', 'gestaureerd'
                     ]

# list of auxiliaries of the sentences
list_auxiliaries = ['hebt', 'hebt', 'hebt', 'hebt', 'hebt', 'hebt', 'wordt', 'wordt', 'wordt', 'wordt',
                     'heeft', 'heeft', 'heeft', 'heeft', 'wordt', 'wordt', 'is', 'is', 'wordt', 'wordt',
                     'is', 'is', 'is', 'is', 'heeft', 'heeft', 'heeft', 'heeft', 'heeft', 'heeft',
                     'is', 'is', 'is', 'is', 'heeft', 'heeft', 'is', 'is', 'heeft', 'heeft',
                     'is', 'is', 'heeft', 'heeft', 'heeft', 'heeft', 'wordt', 'wordt', 'is', 'is',
                     'heeft','heeft','heeft','heeft','heeft','heeft', 'was', 'was', 'was', 'was', 'was'
                     , 'was', 'was', 'was', 'zijn', 'zijn', 'zijn', 'zijn', 'is', 'is',
                     'zijn', 'zijn', 'zijn', 'zijn', 'is', 'is', 'zijn', 'zijn', 'is', 'is', 'is', 'is',
                     'is', 'is', 'is', 'is', 'is', 'is', 'is', 'is', 'is', 'is', 'is', 'is', 'is', 'is',
                     'wordt', 'wordt', 'is', 'is', 'wordt', 'wordt'
                     ]

# create a dictionary for the sentences and their characteristics
dictionary_examples = {'sentence':list_examples, 
                                'participle': list_participles, 
                                'auxiliary': list_auxiliaries, 
                                'participle_index':list_indices, 
                                'order': list_colors}

# create a dataframe from the dictionary
df_examples = pd.DataFrame(dictionary_examples)

# get the PCA transformed dataframe
df_pca = get_df_with_pca(df_examples)

# create a scatterplot for the sentences with the verb orders in red and green
fig = px.scatter(df_pca, x='x', y='y',color=df_pca['order'], hover_data='sentence', color_discrete_sequence=["green", "red"])
fig.show()

# example of how you can visualize certain features
# create a scatterplot with all the sentences with a specific past participle in another color
def custom_color_mapping(participle):
  if participle == 'gewerkt':
    return 'gewerkt'
  else: 
    return 'other verb'

df_pca['participle'] = df_pca['participle'].apply(custom_color_mapping)
color_discrete_map = {'gewerkt': 'rgb(0, 0, 255)', 'other verb': 'rgb(255, 0, 255)'}

fig = px.scatter(df_pca, x='x', y='y', color=df_pca['participle'], color_discrete_map = color_discrete_map, hover_data='sentence')

fig.show()

# function to calculate cosine similarity between sentences
def get_cosine_distance_from_sentences(df1,df2):
  for index, sentence1 in df1.iterrows():
    for index, sentence2 in df2.iterrows():
      sim1 = encode_sentence_and_extract_position_cuda(sentence1["sentence"], sentence1["participle_index"])
      sim2 = encode_sentence_and_extract_position_cuda(sentence2["sentence"], sentence2["participle_index"])
  return cosine_similarity(torch.cat([sim1], dim=0).cpu().detach().numpy(), torch.cat([sim2], dim=0).cpu().detach().numpy())


# example usage of cosine similarity calculation: two last sentences with 'gesloten'
counter = 0
for element in list_participles:
    if element == "gesloten":
        index2 = counter
        index1 = index2
    counter += 1
#print(index1)
#print(index2)
df1 = pd.DataFrame({"sentence":[list_examples[index2-1]],"participle_index":[list_indices[index2-1]]})
df2 = pd.DataFrame({"sentence":[list_examples[index2]],"participle_index":[list_indices[index2]]})

print(get_cosine_distance_from_sentences(df1,df2))
