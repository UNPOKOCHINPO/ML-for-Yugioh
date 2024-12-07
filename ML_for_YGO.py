import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from tqdm import tqdm
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_FOR_NONE = 0

seed = 99

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)



class CardGameDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path, encoding="UTF-8", header=None)
        data.columns = ["先攻・後攻"] + [f"card{i}" for i in range(1, 13)] + ["勝敗", "相手デッキタイプ"]

        all_cards = sorted(set(data.iloc[:, 1:13].stack()))
        self.card2idx = {card: idx for idx, card in enumerate(all_cards, start=1)}
        self.card2idx["n"] = NUM_FOR_NONE

        all_decks = sorted(data["相手デッキタイプ"].dropna().unique())
        self.deck2idx = {deck: idx for idx, deck in enumerate(all_decks, start=1)}
        self.deck2idx["n"] = NUM_FOR_NONE


        self.features = []
        self.labels = []
        for _, row in data.iterrows():
            cards = [self.card2idx.get(row[f"card{i}"], 0) for i in range(1, 13)]
            deck_type_idx = self.deck2idx[row["相手デッキタイプ"]]
            first_turn = 1 if row["先攻・後攻"] == "先攻" else 0
            label = 1 if row["勝敗"] == "勝ち" else 0
            self.features.append([first_turn] + cards + [deck_type_idx])
            self.labels.append(label)

            bl = (len(row)==15)
            bl = bl and (row[-2]=="勝ち" or row[-2]=="負け")
            bl = bl and (row[0]=="先攻" or row[0]=="後攻")


            if not bl:
                print(row.values)
                raise AssertionError

        self.features = torch.tensor(self.features, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class WinPredictionModel(nn.Module):
    def __init__(self, hidden_size, card2idx, deck2idx):
        super(WinPredictionModel, self).__init__()
        self.embedding = nn.Embedding(len(card2idx) + len(deck2idx) + 2, hidden_size) 

        self.conv1 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=4) 
        self.conv2 = nn.Conv1d(hidden_size // 2, hidden_size // 4, kernel_size=3) 
        self.conv3 = nn.Conv1d(hidden_size // 4, hidden_size // 8, kernel_size=2, padding=1) 

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embeddings = self.embedding(x)

        mask = (x[:, 1:] == NUM_FOR_NONE)

        expanded_mask = mask.unsqueeze(2).expand(-1, -1, embeddings.size(2))
        masked_embeddings = embeddings[:, 1:, :]
        masked_embeddings[expanded_mask] = 0
        embeddings[:, 1:, :] = masked_embeddings
        embeddings = embeddings.permute(0, 2, 1)

        conv_output = self.conv1(embeddings)
        conv_output = nn.ReLU()(conv_output)
        conv_output = self.conv2(conv_output)
        conv_output = nn.ReLU()(conv_output)
        conv_output = conv_output.mean(dim=2)
        output = self.fc_layers(conv_output)
        return self.softmax(output)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import random

def train_model(model, train_dataloader, criterion, optimizer, scheduler, epochs=20, num_aug=20):
    model.train()
    print("\nmodel training...")


    
    for epoch in tqdm(range(epochs)):
        all_augmented_features = []
        all_augmented_labels = []

        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)
            batch_augmented_features = []
            batch_augmented_labels = []

            for feature, label in zip(features, labels):
                for _ in range(num_aug):
                    aug_feature = feature.clone()
                    forward = feature[1:6].tolist()
                    random.shuffle(forward)
                    aug_feature[1:6] = torch.tensor(forward).to(device)
                    backward = feature[7:13].tolist()
                    random.shuffle(backward)
                    aug_feature[7:13] = torch.tensor(backward).to(device)
                    batch_augmented_features.append(aug_feature)
                    batch_augmented_labels.append(label)

            all_augmented_features.extend(batch_augmented_features)
            all_augmented_labels.extend(batch_augmented_labels)

        augmented_dataset = list(zip(all_augmented_features, all_augmented_labels))
        random.shuffle(augmented_dataset)
        all_augmented_features, all_augmented_labels = zip(*augmented_dataset)

        augmented_features = torch.stack(all_augmented_features)
        augmented_labels = torch.stack(all_augmented_labels)
        augmented_dataloader = DataLoader(
            TensorDataset(augmented_features, augmented_labels),
            batch_size=train_dataloader.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )

        for batch_features, batch_labels in augmented_dataloader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            batch_labels = batch_labels.long()

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)  
            loss.backward()
            optimizer.step()

        scheduler.step()


import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import random
def evaluate_model(model, test_dataloader, num_aug=20):
    model.eval()
    correct, total = 0, 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in test_dataloader:
            features, labels = features.to(device), labels.to(device)
            majority_predictions = []

            for feature in features:
                augmented_features = []
                for _ in range(num_aug):
                    aug_feature = feature.clone()
                    forward = feature[1:6].tolist()
                    random.shuffle(forward)
                    aug_feature[1:6] = torch.tensor(forward).to(device)
                    backward = feature[7:13].tolist()
                    random.shuffle(backward)
                    aug_feature[7:13] = torch.tensor(backward).to(device)
                    augmented_features.append(aug_feature)

                augmented_features = torch.stack(augmented_features)

                outputs = model(augmented_features)
                _, predictions = torch.max(outputs, 1)

                majority_prediction = predictions.sum().item() > (len(predictions) / 2)
                majority_predictions.append(majority_prediction)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(majority_predictions)

            #majority_predictions = torch.tensor(majority_predictions).to(device)
            #correct += (majority_predictions == labels).sum().item()
            total += labels.size(0)

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    accuracy = (conf_matrix[0,0]+conf_matrix[1,1]) / (conf_matrix[0,0]+conf_matrix[1,1]+conf_matrix[0,1]+conf_matrix[1,0])
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    #print(all_labels, all_predictions)
    f1 = 2*precision*recall/(precision+recall)#f1_score(all_labels, all_predictions)


    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return accuracy, f1
"""
def evaluate_model_with_f1(model, test_dataloader, num_aug=20):


    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in test_dataloader:
            majority_predictions = []

            for feature in features:
                augmented_features = []
                for _ in range(num_aug):
                    aug_feature = feature.clone()
                    
                    forward = feature[1:6].tolist()
                    random.shuffle(forward)
                    aug_feature[1:6] = torch.tensor(forward)
                    backward = feature[7:13].tolist()
                    random.shuffle(backward)
                    aug_feature[7:13] = torch.tensor(backward)
                    augmented_features.append(aug_feature)

                augmented_features = torch.stack(augmented_features).to(device)

                outputs = model(augmented_features)
                _, predictions = torch.max(outputs, 1)
                majority_prediction = predictions.sum().item() > (len(predictions) / 2)
                majority_predictions.append(majority_prediction)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(majority_predictions)


    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)


    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return f1

"""

from lime.lime_tabular import LimeTabularExplainer
import numpy as np


def explain_predictions_with_lime(model, train_dataset, card2idx, deck2idx, num_features=14, enemy_yuhatsu="増G"):
    model.eval()


    feature_names = ["先攻・後攻"] + [f"card{i}" for i in range(1, 13)] + ["相手デッキタイプ"]
    class_names = ["負け", "勝ち"]
    categorical_features = list(range(len(feature_names)))
    explainer = LimeTabularExplainer(
        train_dataset.tensors[0].cpu().numpy(),
        feature_names=feature_names,
        class_names=class_names,
        categorical_features=categorical_features,
        mode='classification'
    )
    
    def predict_fn(inputs):
        inputs = torch.tensor(inputs, dtype=torch.long).to(device)
        outputs = model(inputs)
        return outputs.cpu().detach().numpy()

    exp_dict = {}
    exp_med_dict = {}
    exp_dict_for_decks = {}
    exp_med_dict_for_decks = {}
    with open("deckList.txt","r",encoding="UTF-8") as f:
        lines = f.read().splitlines()
    
    none_mapped = card2idx["n"]
    
    
    random_dataset = [torch.tensor([1]+\
                        [card2idx[random_data] for random_data in random.sample(lines,5)]+\
                        [none_mapped]+\
                        [card2idx[f"{enemy_yuhatsu}"],none_mapped,none_mapped,none_mapped,none_mapped,none_mapped,none_mapped])\
                            for _ in range(1000)]
    
    """understandable_sample1 = torch.tensor([1]+\
                        [card2idx["セアミン"],card2idx["ベイゴマ"],card2idx["泡影"],card2idx["泡影"],card2idx["泡影"]]+\
                        [none_mapped]+\
                        [card2idx["うらら"],none_mapped,none_mapped,none_mapped,none_mapped,none_mapped,none_mapped])
    understandable_sample2 = torch.tensor([1]+\
                        [card2idx["セアミン"],card2idx["キャリー"],card2idx["泡影"],card2idx["泡影"],card2idx["泡影"]]+\
                        [none_mapped]+\
                        [card2idx["うらら"],none_mapped,none_mapped,none_mapped,none_mapped,none_mapped,none_mapped])"""
    
    """for instance in [understandable_sample1, understandable_sample2]:

        #instance = random_dataset[instance_index]
        instance = instance.unsqueeze(0).to(device)
        #print(instance)
        # 説明を生成
        exp = explainer.explain_instance(
            instance.cpu().numpy().flatten(),
            predict_fn,
            num_features=num_features,
            labels=(0,1)
        )
        print("-----")
        for feature_idx, weight in exp.as_list(label=1):
            feat_name = (feature_idx.split("=")[0])
            idx = int(feature_idx.split("=")[1])
            
            if "card" not in feat_name:
                continue
            keys = [key for key, val in card2idx.items() if val == idx]
            data = keys[0]
            print(data, weight)
        print("-----")
    """

    print("generating explanation...")
    for instance_index in tqdm(range(len(random_dataset))):

        instance = random_dataset[instance_index]
        instance = instance.unsqueeze(0).to(device)
        exp = explainer.explain_instance(
            instance.cpu().numpy().flatten(),
            predict_fn,
            num_features=num_features,
            labels=(0,1)
        )


        
        for feature_idx, weight in exp.as_list(label=1):
            feat_name = (feature_idx.split("=")[0])
            idx = int(feature_idx.split("=")[1])

            if "card" in feat_name:
                keys = [key for key, val in card2idx.items() if val == idx]
                data = keys[0]
            elif "デッキ" in feat_name:
                keys = [key for key, val in deck2idx.items() if val == idx]
                data = keys[0]
            elif "先攻" in feat_name:
                data = ("先攻" if idx==1 else "後攻")
            
            if data!="n":
                if "card" in  feat_name:
                    if exp_dict.get(data) is None:
                        exp_dict[data] = []    
                    exp_dict[data].append(weight)
                    exp_dict[data] = sorted(exp_dict[data])

                    if len(exp_dict[data])%2 != 0:
                        exp_med_dict[data] = exp_dict[data][len(exp_dict[data])//2]
                    else:
                        exp_med_dict[data] = (exp_dict[data][len(exp_dict[data])//2]+exp_dict[data][len(exp_dict[data])//2-1])/2
                if "デッキ" in  feat_name:
                    if exp_dict_for_decks.get(data) is None:
                        exp_dict_for_decks[data] = []    
                    exp_dict_for_decks[data].append(weight)
                    exp_dict_for_decks[data] = sorted(exp_dict_for_decks[data])
                    
                    if len(exp_dict[data])%2 != 0:
                        exp_med_dict_for_decks[data] = exp_dict_for_decks[data][len(exp_dict_for_decks[data])//2]
                    else:
                        exp_med_dict_for_decks[data] = (exp_dict_for_decks[data][len(exp_dict_for_decks[data])//2]+exp_dict_for_decks[data][len(exp_dict_for_decks[data])//2-1])/2

    return exp_med_dict

import matplotlib.pyplot as plt
import numpy as np

def plot_importance_of_cards(data,label="増G"):
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

    plt.rcParams["font.family"] = "MS Gothic"
    cmap = plt.get_cmap("coolwarm")

    grid_data = list(sorted_data.values()) + [np.nan for _ in range(((len(sorted_data) + 5 - 1) // 5) * 5 - len(sorted_data))]
    grid_data = np.array(grid_data).reshape((len(sorted_data) + 5 - 1) // 5, 5)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = grid_data
    cax = ax.imshow(colors, cmap=cmap, aspect='auto', vmin=-0.3, vmax=0.3) 

    for i in range((len(sorted_data) + 5 - 1) // 5):
        for j in range(5):
            if not np.isnan(grid_data[i, j]):
                ax.text(j, i, list(sorted_data.keys())[i * 5 + j] + f"\n{sorted_data[list(sorted_data.keys())[i * 5 + j]]:.4f}", ha='center', va='center', color='black', fontsize=6)

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.colorbar(cax, ax=ax, orientation='vertical', label='Weight Value', ticks=np.arange(-0.3, 0.31, 0.1))

    plt.title('Importance of cards')
    plt.savefig(f"explanation_{label}.png")



if __name__=='__main__':

    accs = []
    f1_scores = []
    max_acc = 0
    for _ in range(12):
        file_path = "battle_data.csv"
        dataset = CardGameDataset(file_path)

        features = dataset.features.numpy()
        labels = dataset.labels.numpy()
        
        np.random.seed(seed)
        random.seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=123)

        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32))
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,worker_init_fn=seed_worker,generator=g)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False,worker_init_fn=seed_worker,generator=g)



        hidden_size = 64
        model = WinPredictionModel(hidden_size, dataset.card2idx, dataset.deck2idx).to(device)

        def init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        model.apply(init_weights)


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)

        num_aug = 20
        train_model(model, train_dataloader, criterion, optimizer, scheduler, epochs=30, num_aug=num_aug)
        accuracy, f1_score = evaluate_model(model, test_dataloader, num_aug=num_aug)
        accs.append(accuracy)

        #f1_score = evaluate_model_with_f1(model, test_dataloader, num_aug=num_aug)
        f1_scores.append(f1_score)

        if max_acc < accuracy:
            for yuhatsu in ["n","増G","うらら","泡影"]:
                if accuracy > 0.8:
                    score_dict = explain_predictions_with_lime(model, train_dataset, dataset.card2idx, dataset.deck2idx, enemy_yuhatsu=yuhatsu)
                    plot_importance_of_cards(score_dict,label=yuhatsu)
                max_acc = accuracy

    print(f"Accuracy: {100*sum(accs)/len(accs):.2f}%")
    print(f"F1-score: {100*sum(f1_scores)/len(f1_scores):.2f}%")