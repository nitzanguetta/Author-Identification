import pickle
import re
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# nltk.download('stopwords')
# nltk.download('vader_lexicon')

embedding_file = "train_from_GoogleNews-vectors-negative300.pkl"
train_file = "trump_train.tsv"
test_file = "trump_test.tsv"

time_key_format = '%Y-%m-%d %H:%M:%S'
time_key = 'time'
normalized_text_key = 'processed_text'
id_key = 'id'
user_handle_key = 'user_handle'
text_key = 'text'
device_key = 'device'
label_key = 'label'

########################################################################################################################

""" General Functions """


def get_file_df(file_name, predict=False):
    if not predict:
        df = pd.read_csv(file_name, sep="\t", header=None, quoting=csv.QUOTE_NONE,
                         names=[id_key, user_handle_key, text_key,
                                time_key, device_key]).dropna(subset=[time_key])
        df[label_key] = df.apply((lambda row: 0 if (row[device_key] == 'android' and
                                                    row[user_handle_key] == 'realDonaldTrump' and not
                                                           row[text_key].startswith('RT')) else 1), axis=1)
    else:
        df = pd.read_csv(file_name, sep="\t", header=None, quoting=csv.QUOTE_NONE, names=[user_handle_key,
                                                                                          text_key,
                                                                                          time_key])

    df[time_key] = df[time_key].apply(lambda x: datetime.strptime(x, time_key_format))
    df[normalized_text_key] = df[text_key].apply(normalize_text)
    return df


def get_file_X_y():
    df = get_file_df(train_file).sort_values([time_key]).reset_index()
    return df.drop([label_key], axis=1), df[label_key]


def normalize_text(text):
    """Returns a normalized string based on the specifiy string.
       You can add default parameters as you like (they should have default values!)
       You should explain your decitions in the header of the function.

       the returned normalized text is clean from stopwords (by nltk library).

       Args:
           text (str): the text to normalize

       Returns:
           string. the normalized text.
    """
    words = re.compile(r'''(?x)(?:[A-Z]\.)+ | \w+(?:-\w+)* | \.\.\. | \$?\d+(?:\.\d+)?%? | [][.,;"'?():_`-]''').findall(text.lower())
    return ' '.join([word for word in words if word not in set(stopwords.words('english'))])


def print_report(y_labels, y_pred, clf):
    fpr, tpr, thresholds = roc_curve(y_labels, y_pred)
    cls_auc_score = auc(fpr, tpr)
    print(f"##### classifier {clf} report: #####")
    print(confusion_matrix(y_labels, y_pred))
    print(classification_report(y_labels, y_pred))
    print(f"accuracy: {accuracy_score(y_labels, y_pred)}")
    print(f"auc: {cls_auc_score}")


def get_train_test_by_split_index(features, split_idx, y):
    train_x, test_x = features[:split_idx], features[split_idx:]
    train_y, test_y = y[:split_idx], y[split_idx:]
    return test_x, test_y, train_x, train_y

########################################################################################################################

"""Machine learning"""


def tokenizer(text):
    return [item for item in nltk.word_tokenize(text) if item not in set(stopwords.words('english'))]


def get_feature_union():
    return FeatureUnion(transformer_list=[
                ('tfidf', Pipeline([
                    ('selector', ItemSelector(key=normalized_text_key)),
                    ('TfidfVectorizer',
                     TfidfVectorizer(lowercase=False, min_df=1, tokenizer=tokenizer, max_features=9000,
                                     ngram_range=(1, 3)))])),
                ('weekday', Pipeline([
                    ('selector', ItemSelector(key=time_key)),
                    ('Weekday', Weekday())])),
                ('elections', Pipeline([
                    ('selector', ItemSelector(key=time_key)),
                    ('Elections', Elections())])),
                ('hour', Pipeline([
                    ('selector', ItemSelector(key=time_key)),
                    ('Hour', Hour())])),
                ('sentiment', Pipeline([
                    ('selector', ItemSelector(key=text_key)),
                    ('SentimentAnalysis', SentimentAnalysis())])),
                ('kmeans', Pipeline([
                    ('selector', ItemSelector(key=normalized_text_key)),
                    ('WordEmbedding', WordEmbedding()),
                    ('KMeans', KMeans(n_clusters=4))
                ]))
            ])


class ItemSelector(BaseEstimator, TransformerMixin):
    # from https://scikit-learn.org/0.18/auto_examples/hetero_feature_union.html#
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, df, y=None):
        return df[self.key]


class Weekday:
    def fit(self, df_threads, y=None):
        return self

    def transform(self, df, y=None):
        return np.array([time.weekday() / 7 for time in df]).reshape(-1, 1)


class Elections:
    def fit(self, df_threads, y=None):
        return self

    def transform(self, df, y=None):
        elections_date = datetime.strptime("8-11-2016", '%d-%m-%Y')
        return np.array(list(map(lambda date: 1 if date > elections_date else 0, df))).reshape(-1, 1)


class Hour:
    def fit(self, df_threads, y=None):
        return self

    def transform(self, df, y=None):
        hour_list = [time.hour for time in df]
        hour_unique = list(set(hour_list))
        return np.array([hour_unique.index(hour) / len(hour_unique) for hour in hour_list]).reshape(-1, 1)


class SentimentAnalysis:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        df_normalized_text = df.apply((lambda text: re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text).lower()))
        scores = [self.analyzer.polarity_scores(text) for text in df_normalized_text]
        return np.array([list(score.values()) for score in scores])


class WordEmbedding:
    def __init__(self):
        self.embeddings = pickle.load(open(embedding_file, 'rb'))
        self.embedding_size = 300

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        embeddings_average = []
        df_text = df.apply(lambda tweet: tweet if type(tweet) is str else "")
        for text in df_text:
            tokens = text.split()
            tokens_num = len(tokens)
            tokens_num = 1 if (tokens_num == 0) else tokens_num
            embeddings_sum = np.array([float(0)] * self.embedding_size)
            for token in tokens:
                if token in self.embeddings:
                    embeddings_sum += np.array(self.embeddings[token])
            embeddings_average.append(embeddings_sum / tokens_num)
        return embeddings_average


########################################################################################################################


"""Deep learning"""


class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, matrix=None):
        super(RNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        if matrix is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(matrix))

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.sig(out)
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout, matrix=None):
        super(CNN, self).__init__()
        if matrix is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(matrix))

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters,
                                kernel_size=(filter_sizes[2], embedding_dim))
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        x = x.long()
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim=1))
        out = self.fc(cat)
        sig_out = self.sig(out)
        return sig_out, hidden

    def init_hidden(self, batch_size):
        return None


def get_features(df):
    embeddings = pickle.load(open(embedding_file, 'rb'))
    embedding_size = 300
    embeddings_average = []
    df_text = df.apply(lambda tweet: tweet if type(tweet) is str else "")
    for text in df_text:
            tokens = text.split()
            tokens_num = len(tokens)
            tokens_num = 1 if (tokens_num == 0) else tokens_num
            embeddings_sum = np.array([float(0)] * embedding_size)
            for token in tokens:
                if token in embeddings:
                    embeddings_sum += np.array(embeddings[token])
            embeddings_average.append(embeddings_sum / tokens_num)
    return np.array(embeddings_average)


def word_encoding(train_tweets, tweets):
    reviews_split = train_tweets
    all_text = ' '.join(reviews_split)
    # create a list of words
    words = all_text.split()

    new_reviews = []
    for review in np.array(tweets):
        review = review.split()
        new_text = []
        for word in review:
            if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
                if word not in words:
                    new_text.append('unk')  # unknown words are mapped into 'unk' encoding value
                else:
                    new_text.append(word)
        new_reviews.append(new_text)

    # word_embedding_file
    counts = Counter(words)
    counts['unk'] = -1  # unknown words are mapped into 'unk' encoding value
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    ## use the dict to tokenize each review in reviews_split
    ## store the tokenized reviews in reviews_ints
    reviews_ints = []
    for review in new_reviews:
        reviews_ints.append([vocab_to_int[word] for word in review])
    return reviews_ints, len(vocab_to_int), vocab_to_int


def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    '''
    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]
    return features

def create_train_weight_matrix(vocab_size, word_to_index):
    word_to_embeddings = pickle.load(open(embedding_file, 'rb'))
    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size+1, 300))
    for word, i in word_to_index.items():
        embedding_vector = word_to_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.zeros(300)
    return embedding_matrix


def test_nn(batch_size, criterion, net, test_loader):
    # Get test data loss and accuracy
    test_losses = []
    num_correct = 0
    h = net.init_hidden(batch_size)
    net.eval()
    first = True
    for inputs, labels in test_loader:
        h = None if h is None else tuple([each.data for each in h])
        output, h = net(inputs, h)

        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        pred = torch.round(output.squeeze())
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy())
        num_correct += np.sum(correct)

        if first:
            y_pred = pd.DataFrame(pred.detach().numpy())
            y_labels = pd.DataFrame(labels.float())
            first = False
        else:
            y_pred = pd.concat([y_pred, pd.DataFrame(pred.detach().numpy())], ignore_index=True)
            y_labels = pd.concat([y_labels, pd.DataFrame(labels.float())], ignore_index=True)
    return y_labels, y_pred


def train_nn(batch_size, clip, counter, criterion, epochs, net, optimizer, print_every, train_loader, valid=False,
             valid_loader=None):
    for e in range(epochs):
        h = net.init_hidden(batch_size)
        for inputs, labels in train_loader:
            counter += 1
            h = None if h is None else tuple([each.data for each in h])
            net.zero_grad()
            output, h = net(inputs, h)
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            if counter % print_every == 0:
                if valid:
                    validate_nn(batch_size, criterion, net, valid_loader)
                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()))


def validate_nn(batch_size, criterion, net, valid_loader):
    val_h = net.init_hidden(batch_size)
    val_losses = []
    net.eval()
    for inputs, labels in valid_loader:
        val_h = None if val_h is None else tuple([each.data for each in val_h])
        output, val_h = net(inputs, val_h)
        val_loss = criterion(output.squeeze(), labels.float())
        val_losses.append(val_loss.item())


def get_train_test_loaders(batch_size, features, split_idx, y, valid=False):
    valid_loader = None
    test_x, test_y, train_x, train_y = get_train_test_by_split_index(features, split_idx, y)
    if valid:
        test_idx = int(len(test_x) * 0.5)
        val_x, test_x = test_x[:test_idx], test_x[test_idx:]
        val_y, test_y = test_y[:test_idx], test_y[test_idx:]
        valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
        valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size, drop_last=True)

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size, drop_last=True)

    return test_loader, train_loader, valid_loader


def get_features_and_matrix(split_idx, tweets):
    reviews_ints, len_unique_words, vocab_to_int = word_encoding(tweets[:split_idx], tweets)
    seq_length = np.max([len(x) for x in reviews_ints])
    features = pad_features(reviews_ints, seq_length=seq_length)
    matrix = create_train_weight_matrix(len_unique_words, vocab_to_int)
    return features, len_unique_words, matrix

########################################################################################################################


""" Training and Testing """


def train_and_test_machine_learning_models_cv():
    X, y = get_file_X_y()
    phase = 0
    tscv = TimeSeriesSplit(n_splits=2)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        classifiers = {
            'LogisticRegression': LogisticRegression(solver='liblinear'),
            'randomForest': RandomForestClassifier(n_jobs=2, random_state=0),
            'SVC_rbf': SVC(kernel='rbf'),
            'SVC_linear': SVC(kernel='linear')
        }
        for clf_name, clf in classifiers.items():
            pipeline = Pipeline([('features', get_feature_union()), ('clf', clf)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            print_report(y_test, y_pred, clf)
        phase += 1


def train_and_test_machine_learning_models_20_80():
    X, y = get_file_X_y()
    split_idx = int(len(X[normalized_text_key]) * 0.8)
    X_test, y_test, X_train, y_train = get_train_test_by_split_index(X, split_idx, y)
    classifiers = {
        'LogisticRegression': LogisticRegression(solver='liblinear'),
        'randomForest': RandomForestClassifier(n_jobs=2, random_state=0),
        'SVC_rbf': SVC(kernel='rbf'),
        'SVC_linear': SVC(kernel='linear')
    }
    for clf_name, clf in classifiers.items():
        pipeline = Pipeline([('features', get_feature_union()), ('clf', clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print_report(y_test, y_pred, clf)


def train_test_final_rnn():
    X, y = get_file_X_y()
    y = np.array(y)
    tweets = X[normalized_text_key]
    split_idx = int(len(tweets) * 0.8)
    batch_size = 50
    output_size = 1
    n_layers = 2
    lr = 0.1
    embedding_dim = 300
    clip = 3
    hidden_dim = 32
    epochs = 10
    counter = 0
    print_every = 100

    features, len_unique_words, matrix = get_features_and_matrix(split_idx, tweets)
    test_loader, train_loader, _ = get_train_test_loaders(batch_size, features, split_idx, y)
    vocab_size = len_unique_words + 1

    print(f"lr:{lr}, clip:{clip}, hidden_dim:{hidden_dim}, embedding_dim:{embedding_dim}")
    net = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, matrix=matrix)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_nn(batch_size, clip, counter, criterion, epochs, net, optimizer, print_every, train_loader)
    y_labels, y_pred = test_nn(batch_size, criterion, net, test_loader)
    print_report(y_labels, y_pred, type(net).__name__)


def train_test_rnn():
    import itertools
    hyperparameters = {'lr': [0.1, 0.001, 0.0001], 'clip': [1, 3, 5], 'hidden_dim': [32, 64, 128],
                       'embedding_dim': [200, 300, 400]}
    X, y = get_file_X_y()
    y = np.array(y)
    tweets = X[normalized_text_key]
    split_idx = int(len(tweets) * 0.8)
    batch_size = 50
    output_size = 1
    n_layers = 2
    embedding_dim = 300
    epochs = 10
    counter = 0
    print_every = 100

    features, len_unique_words, matrix = get_features_and_matrix(split_idx, tweets)
    test_loader, train_loader, valid_loader = get_train_test_loaders(batch_size, features, split_idx, y, valid=True)

    vocab_size = len_unique_words + 1  # +1 for the 0 padding + our word tokens

    for lr, clip, hidden_dim in list(
            itertools.product(hyperparameters['lr'], hyperparameters['clip'], hyperparameters['hidden_dim'])):
        print(f"lr:{lr}, clip:{clip}, hidden_dim:{hidden_dim}, embedding_dim:{embedding_dim}")
        classifiers = {'RNN': (RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, matrix=matrix),
                               nn.BCELoss()),
                      'CNN': (CNN(vocab_size, embedding_dim, 100, [3, 4, 5], 1, 0.5, matrix=matrix),
                              nn.BCEWithLogitsLoss())}

        for net, criterion in classifiers.values():
            optimizer = torch.optim.Adam(net.parameters(), lr=lr)
            train_nn(batch_size, clip, counter, criterion, epochs, net, optimizer, print_every, train_loader,
                     valid=True, valid_loader=valid_loader)
            y_labels, y_pred = test_nn(batch_size, criterion, net, test_loader)
            print_report(y_labels, y_pred, type(net).__name__)

########################################################################################################################


""" Driver Functions"""


def load_best_model():
    """
    Returns the best model, classifying tweets text as Trump/ not Trump.
    """
    return pickle.load(open("best_model_307937169.pkl", 'rb'))


def train_best_model():
    """
    Training a classifier (the same classifier and parameters returned by load_best_model()) from scratch.
    Returns a pipeline.
    """
    X, y = get_file_X_y()
    pipeline = Pipeline([('features', get_feature_union()), ('clf', SVC(kernel='linear'))])
    pipeline.fit(X, y)
    return pipeline


def predict(m, fn):
    """Returns the predictions of model m on test set from the path fn
       Args:
           m: A trained model
           fn: the full path to a file in the same format as the test set

       Returns:
           list. containing of 0s and 1s, corresponding to the lines in the specified file.
    """
    test_df = get_file_df(fn, True)
    test_df[normalized_text_key] = test_df[text_key].apply(normalize_text)
    return m.predict(test_df)


# if __name__ == "__main__":
    # train_and_test_machine_learning_models_20_80()
    # train_and_test_machine_learning_models_cv()
    # train_test_rnn()  # to find the best parameters for rnn.
    # train_test_final_rnn()  # run rnn with the chosen params.
    #
    # pipeline = train_best_model()
    # pickle.dump(pipeline, open("best_model_307937169.pkl", "wb"))
    # print(load_best_model())
    # pipeline = load_best_model()
    # y_pred = predict(pipeline, 'trump_test.tsv')
    # y_pred_str = " ".join([str(i) for i in y_pred])
    # with open("307937169.txt", "w") as text_file:
    #     text_file.write(y_pred_str)

