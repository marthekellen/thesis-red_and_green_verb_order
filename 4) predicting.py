# install necessary libraries
!pip install accelerate -U
!pip install transformers
!pip install scikit-learn

# import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from urllib.request import urlretrieve
from scipy.special import softmax
import copy
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# define the GPU
cuda = torch.device('cuda')

# Load tokenizer and model from pre-trained BERTje or RoBBERT model
tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
model = AutoModelForSequenceClassification.from_pretrained("GroNLP/bert-base-dutch-cased")
#transformer_model = 'pdelobelle/robbert-v2-dutch-base'
#tokenizer = AutoTokenizer.from_pretrained(transformer_model)
#model = AutoModelForSequenceClassification.from_pretrained(transformer_model)

# move model to GPU
model.cuda()

# Download the dataset from the CCL server
FILENAME = "RoodGroenAnthe.csv"
DATASET_URL = f"https://www.ccl.kuleuven.be/SONAR/{FILENAME}"
urlretrieve(DATASET_URL, FILENAME)

# Read dataset with pandas
df = pd.read_csv(FILENAME)
df.head()

# Define custom dataset class for handling encodings and labels
class RoodGroenDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Function to create dataset (for each sentence in the dataset, we create both the red and the green verb order)
def create_set(df):
    sentences = []
    labels = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        part_idx = row['participle_index']
        aux_idx = row['auxiliary_index']
        sentence = row['sentence']
        sent_words = sentence.split(' ')
        sent_words_green = copy.deepcopy(sent_words)
        sent_words_red = copy.deepcopy(sent_words)

        if aux_idx > part_idx:
            sent_words_red[part_idx] = row['auxiliary']
            sent_words_red[aux_idx] = row['participle']
        elif part_idx > aux_idx:
            sent_words_green[part_idx] = row['auxiliary']
            sent_words_green[aux_idx] = row['participle']
        sentences.append((' '.join(sent_words_green), ' '.join(sent_words_red)))
        labels.append(0 if row['order'] == "green" else 1)

    encodings = tokenizer(sentences, truncation=True, padding=True, max_length=128)
    dataset = RoodGroenDataset(encodings, labels)
    return dataset

# Sample datasets for testing
df_small = df.sample(frac=0.05)
df_medium = df.sample(frac=0.1)
df_large = df.sample(frac=0.15)
df_larger = df.sample(frac=0.20)
df_largest = df.sample(frac=0.25)

df_small.shape[0]
df_medium.shape[0]
df_large.shape[0]
df_larger.shape[0]
df_largest.shape[0]

# Create train, validation, and test sets
train_df, test_valid_df = train_test_split(df_largest, test_size=0.4, random_state=42)
test_df, valid_df = train_test_split(test_valid_df, test_size=0.5, random_state=42)

train_dataset = create_set(train_df)
valid_dataset = create_set(valid_df)
test_dataset = create_set(test_df)

# Define metrics computation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }



# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    learning_rate=5e-5,
    fp16=True,
    evaluation_strategy='steps',
    eval_steps=100,
)

# Initialize Trainer
trainer = Trainer(
    model=model,                         # model with classification head
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,             # evaluation dataset
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Make predictions on the test dataset
pred = trainer.predict(test_dataset)

# Print prediction matrix
pred.metrics

# Function to calculate confusion metrics
def compute_confusion_matrix(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    conf_matrix = confusion_matrix(labels, preds)
    return conf_matrix

# Print confusion metrics
compute_confusion_matrix(pred)

# using the softmax function, we transform the logits to probabilities
probs = softmax(pred.predictions, axis=1)
# then we sort the sentences according to the probabilities
sorted_sentences = np.argsort(probs[:,0])

# show 100 sentences with highest green probability
df_prob_green = test_df.iloc[sorted_sentences[-100:]]
test_df.iloc[sorted_sentences[-100:]]

# show 100 sentences with highest red probability for the past participle 'geschreven'
df_schrijven = df[df['participle_lemma'] == 'schrijven']
pred = trainer.predict(create_set(df_schrijven))

# using the softmax function, we transform the logits to probabilities
probs = softmax(pred.predictions, axis=1)
# then we sort the sentences according to the probabilities
sorted_sentences = np.argsort(probs[:,0])

#print(df_schrijven)
df_prob_red = df_schrijven.iloc[sorted_sentences[0:100]]
df_schrijven.iloc[sorted_sentences[0:100]]

# save the 250 sentences with the highest probabilty of accuring in the red verb order according to BERTje
#df_prob_red.to_csv('df_prob_red__schrijven_B250.csv')


# misclassifications for the past participle 'geschreven'
# in order to get the misclassifications for the verb 'geschreven',
# we use the dataframe with the sentences containing 'geschreven' 
# as a testset
df_prob_green = df_schrijven.iloc[sorted_sentences[-600:]]

df_prob_green = df_schrijven.iloc[sorted_sentences[-600:]]
df_red_misclassified = df_prob_green[df_prob_green['order'] == 'red']

# Save misclassified red sentences to a CSV file
df_red_misclassified.to_csv('df_prob_red_misclassified_schrijven_.csv')
