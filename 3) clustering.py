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
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import homogeneity_completeness_v_measure
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc 
from sklearn.cluster import AgglomerativeClustering

# load tokenizer and model from pre-trained BERTje model
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = AutoModel.from_pretrained("GroNLP/bert-base-dutch-cased")

# move model to GPU
cuda = torch.device('cuda')
model.cuda()

# disable gradient calculations to save memory (as we are not training the model)
@torch.no_grad()
def encode_sentence_and_extract_position(sentence, position):
    ids = tokenizer.encode(sentence)
    bert_output = model.forward(torch.tensor(ids, device=cuda).unsqueeze(0),
                                encoder_hidden_states = True)
    final_layer_embeddings = bert_output['last_hidden_state'].squeeze()
    return final_layer_embeddings[position].unsqueeze(0)

# download the dataset from the CCL server
FILENAME = "RoodGroenAnthe.csv"
DATASET_URL = f"https://www.ccl.kuleuven.be/SONAR/{FILENAME}"
urlretrieve(DATASET_URL, FILENAME)

# read dataset with pandas
df = pd.read_csv(FILENAME)
df.head()

# function to filter sentences by lemma
def get_sentences_by_lemma(lemma):
  return df[df["participle_lemma"] == lemma] 

# function to get embeddings from sentences
def get_embeddings_from_sentences(sentences):
  embeddings = []
  ids = []
  # encode all sentences to embeddings
  for sentence_id, sentence in sentences.iterrows():
    try:
      embeddings.append(encode_sentence_and_extract_position(sentence["sentence"], sentence["participle_index"]))
    # print sentences id if there is an error
    except:
      print(sentence["sentence_id"])
      ids.append(sentence["sentence_id"])
  return embeddings, ids

# function to create a new datafram with PCA values
def get_df_with_pca(sentences, n_sample=False, random_state=42):
  # sample a subset of the sentences is specified
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

  # print(pca.explained_variance_ratio_[:15])
  # print(pca.singular_values_[:15])
  
  # insert PCA components in dataframe
  df.insert(1, 'x', components[:, 0])
  df.insert(2, 'y', components[:, 1])
  df.insert(3, 'z', components[:, 2])

  return df

# function to indicate all sentences of a given cluster
def find_cluster(lijst, cluster_index):
    result = []
    for element in lijst:
        if element[0] == cluster_index:
            result.append("cluster")
        else:
            result.append("no_cluster")
    return result

# function to evaluate clustering methods/techniques
def evaluate(lijst1, lijst2):
    ARI = metrics.adjusted_rand_score(lijst1, lijst2)
    Vmeasure = metrics.v_measure_score(lijst1, lijst2)
    FM = metrics.fowlkes_mallows_score(lijst1, lijst2)

    print("ARI: " + str(ARI))
    print("Vmeasure: " + str(Vmeasure))
    print("Fowlkes_mallows: " + str(FM))

# function to calculate cluster purity
def cluster_purity(cluster_labels, true_labels, cluster_index):
    cluster_instances = [true_labels[i] for i, label in enumerate(cluster_labels) if label == cluster_index]
    most_frequent_class_count = max(Counter(cluster_instances).values())
    purity = most_frequent_class_count / len(cluster_instances)
    return purity

# function to calculate cluster purity, for all clusters instead of for one
def cluster_purities(cluster_labels, true_labels):
    cluster_purities = {}
    unique_clusters = set(cluster_labels)
    for cluster_index in range(max(unique_clusters) + 1):
        cluster_instances = [true_labels[i] for i, label in enumerate(cluster_labels) if label == cluster_index]
        class_counts = Counter(cluster_instances)
        most_frequent_class_count = class_counts.most_common(1)[0][1] if class_counts else 0
        purity = most_frequent_class_count / len(cluster_instances) if len(cluster_instances) > 0 else 0
        cluster_purities[cluster_index] = purity
    return cluster_purities

# example usage
example_sentences = get_sentences_by_lemma("schrijven")
df_example = get_df_with_pca(example_sentences)


# set the data to the correct values
copy_sc = example_sentences.copy()
embeddings, _ = get_embeddings_from_sentences(copy_sc)
emb_matrix = torch.cat(embeddings, dim=0)
matrix_np = emb_matrix.cpu().numpy()


# elbow method to determine optimal number of clusters
k_values = range(1, 20)
wcss = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(matrix_np)
    wcss.append(kmeans.inertia_)

# plot results Elbow method
data = {'K': k_values, 'WCSS': wcss}
df_2 = pd.DataFrame(data)
fig = px.scatter(df_2, x='K', y='WCSS', title='Elbow Method for Optimal K',
                 labels={'K': 'Number of Clusters (K)', 'WCSS': 'Within-Cluster Sum of Squares (WCSS)'},
                 template='plotly_white')


# K-means clustering first executed for 2 clusters, and then for 3 clusters (and eventually 7)

# K-means clustering with 2 clusters
kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans_2.fit(matrix_np)
centers = kmeans_2.cluster_centers_


# assign new colors based on clustering results for K-means with 2 clusters
lijst = list(zip(kmeans_2.labels_, df_example['order']))
lijst3 = list(df_example["order"])
lijst4 = []
counter = 0
for el in lijst:
    lijst4.append(lijst3[counter] + "_" + str(el[0]))
    counter += 1

df_example["new_color"] = lijst4

# plot the data with the colored clusters
fig = px.scatter(df_example, x='x', y='y', color='new_color',
                 color_discrete_map={'red_1': 'yellow', 'green_1': 'orange', 'red_0': 'purple', 'green_0': "magenta"},
                 hover_data='sentence')
fig.show()

# evaluation of clustering with 2 clusters
eval_2clus_pred = []
eval_2clus_actual = []
# for the selected verbs, store the actual orders in verbs_eval2
for element in lijst3:
    if element == "red":
        eval_2clus_actual.append(1)
    if element == "green":
        eval_2clus_actual.append(0)
# store the cluster number for each verb in verbs_eval1
for element in lijst:
    eval_2clus_pred.append(element[0])


# evaluate the clustering with the actual verb orders
evaluate(eval_2clus_pred, eval_2clus_actual)

homogeneity, completeness, _ = homogeneity_completeness_v_measure(eval_2clus_pred, eval_2clus_actual)

print("Homogeneity:", homogeneity)
print("Completeness:", completeness)


# example usage cluster purity: calculate purity for the cluster with the index in question
cluster_index = 1
purity = cluster_purity(eval_2clus_pred, eval_2clus_actual, cluster_index)
print("Purity for cluster {}: {:.2f}".format(cluster_index, purity))



# K-means clusters with 3 clusters
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init="auto")
kmeans_3.fit(matrix_np)
centers = kmeans_3.cluster_centers_


# assign new colors based on clustering results for K-means with 3 clusters
lijst = list(zip(kmeans_3.labels_, df_example['order']))
lijst3 = list(df_example["order"])
lijst4 = []
counter = 0
for el in lijst:
    if el[0]==1 and lijst3[counter]=="green":
        lijst4.append("1_green")
    elif el[0]==1 and lijst3[counter]=="red":
        lijst4.append("1_red")
    elif el[0]==2 and lijst3[counter]=="green":
        lijst4.append("2_green")
    elif el[0]==2 and lijst3[counter]=="red":
        lijst4.append("2_red")
    elif el[0]==0 and lijst3[counter]=="green":
        lijst4.append("0_green")
    else:
        lijst4.append("0_red")
    counter += 1
    pass
df_example["new_color"] = lijst4

# plot the data with the colored clusters
fig = px.scatter(df_example, x='x', y='y', color='new_color',
                 color_discrete_map={'red_1': 'yellow', 'green_1': 'orange','red_0': 'purple', 'green_0': "magenta",'red_2': 'blue', 'green_2': "black"},
                 hover_data='sentence')
fig.show()

# evaluation score for clustering with 3 clusters
eval_3clus_pred = []
eval_3clus_actual = []
testlijstscores = []
# in the assumption that cluster 0 is composed of all red verbs,
# the elements from cluster 0 are stored in n
for element in lijst:
    if element[0] == 2 or element[0] == 1:
        eval_3clus_pred.append(1)
    else:
        eval_3clus_pred.append(0)
    eval_3clus_actual.append(element[1])
    testlijstscores.append(element[0])

evaluate(eval_3clus_pred, eval_3clus_actual)

# example for clustering with more than 3 clusters (7 clusters in this example)
kmeans_7 = KMeans(n_clusters=7, random_state=42, n_init="auto")
kmeans_7.fit(matrix_np)
centers = kmeans_7.cluster_centers_

lijst = list(zip(kmeans_7.labels_, df_example['order']))


counters = {'green': [0, 0, 0, 0, 0, 0, 0], "red": [0, 0, 0, 0, 0, 0, 0]}
for tuple in lijst:
    counters[tuple[1]][tuple[0]] += 1

# print the number of sentences in the red and green verb order per cluster
for index in range(7):
    print('cluster 0:' + str(counters["red"][index]) + " " + str(counters["green"][index]))

# plot the red cluster
df_example["new_color"] = find_cluster(lijst, 6)

fig = px.scatter(df_example, x='x', y='y', color='new_color',
                 color_discrete_map={'cluster': 'purple', 'no_cluster': 'yellow'},
                 hover_data='sentence')
fig.show()

homogeneity, completeness, _ = homogeneity_completeness_v_measure(eval_3clus_pred, eval_3clus_actual)

print("Homogeneity:", homogeneity)
print("Completeness:", completeness)


# example usage to find cluster purities for all clusters
purity_per_cluster = cluster_purities(testlijstscores, eval_3clus_actual)
for cluster_index, purity in purity_per_cluster.items():
    print("Purity for cluster {}: {:.2f}".format(cluster_index, purity))


# next clustering method: Agglomerative clustering
# normalize data for Agglomerative clustering
data_scaled = pd.DataFrame(normalize(matrix_np))
data_scaled.head()

# plot dendrogram
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms") 
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))

# Agglomerative Clustering with 2 clusters
aggl_2 = AgglomerativeClustering(n_clusters=2, linkage='ward')
aggl_2.fit_predict(matrix_np)

# assign colors to clusters
colors_hierarchh = []
colors = list(df_example["order"])
aggl_lijstrg = []
aggl_ref = []

for counter in range(len(aggl_2.labels_)):
    aggl_lijstrg.append(str(aggl_2.labels_[counter]))
    if colors[counter] == "red":
        colors_hierarchh.append("r" + str(aggl_2.labels_[counter]))
        aggl_ref.append('1')
    else:
        colors_hierarchh.append("g" + str(aggl_2.labels_[counter]))
        aggl_ref.append('0')

df_example["newcolors"] = colors_hierarchh

fig = px.scatter(df_example, x='x', y='y', color=colors_hierarchh,
                 color_discrete_map={'r1': 'purple', 'g1': 'magenta', 'r0': 'yellow', 'g0': "orange"},
                 hover_data='sentence')
fig.show()

# evaluation of clustering with two clusters
evaluate(aggl_lijstrg, aggl_ref)

# example: more than three clusters
cluster = AgglomerativeClustering(n_clusters=7, linkage='ward', compute_distances=True)  
cluster.fit_predict(data_scaled)

lijst = list(zip(cluster.labels_, df_example['order']))


lijst7redgreen = {"red": [0, 0, 0, 0, 0, 0, 0], "green": [0, 0, 0, 0, 0, 0, 0]}
for element in lijst:
    lijst7redgreen[element[1]][element[0]] += 1

# print the number of sentences in the red and green verb order per cluster
print(lijst7redgreen["red"])
print(lijst7redgreen["green"])


# plot the red clusters
print(lijst)
df_example["new_color"] = find_cluster(lijst, 2)

fig = px.scatter(df_example, x='x', y='y', color='new_color',
                 color_discrete_map={'cluster': 'purple', 'no_cluster': 'yellow'},
                 hover_data='sentence')
fig.show()
