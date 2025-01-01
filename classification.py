import os
import re
import json
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

##### Load data. #####
def load_and_group_labels(directory_path):
    hotel_dataset = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                buffer = ""
                for line in file:
                    stripped_line = line.strip()
                        
                    # Skip comments that start with '//'
                    if stripped_line.startswith("//"):
                        continue
                    
                    # Accumulate lines in buffer until a full JSON object is found
                    buffer += stripped_line
                    
                    try:
                        data = json.loads(buffer)
                        hotel_dataset.append(data)
                        buffer = ""  # Clear buffer once a full JSON object is processed
                    except json.JSONDecodeError:
                        continue

    return hotel_dataset

data = load_and_group_labels("./data") # Load data from your directory.

##### Insert tokens. #####
def insert_aspect_tokens(data):
    result = []

    for idx, entry in enumerate(data):
        text = entry['text']
        labels = entry['labels']

        if len(text) > 512:
            print(f"Skipping entry {idx} due to text length exceeding 512 characters.")
            continue
        
        # Skip entries with empty labels
        if not labels:
            print(f"Skipping entry {idx} due to empty labels.")
            continue
        
        modified_text = text
        category_list = []
        category_subcategory_list = []
        
        for aspect_info in labels:
            aspect = aspect_info['aspect']
            opinion = aspect_info['opinion']
            category = aspect_info['category']
            subcategory = aspect_info['subcategory']

            category_list.append(category)
            combined_label = f"{category}_{subcategory}"
            category_subcategory_list.append(combined_label)
            
            # 找到所有aspect出現的位置
            aspect_positions = [m.start() for m in re.finditer(re.escape(aspect), modified_text)]
            # 找到opinion出現的位置
            opinion_position = modified_text.find(opinion)
            
            if aspect_positions and opinion_position != -1:
                # 找出距離opinion最近的aspect
                closest_aspect_position = min(aspect_positions, key=lambda pos: abs(pos - opinion_position))
                
                # 標記離opinion最近的aspect
                modified_text = (
                    modified_text[:closest_aspect_position] +
                    f"<asp>{aspect}</asp>" +
                    modified_text[closest_aspect_position + len(aspect):]
                )
        
        # 檢查 <asp> 的數量是否與 category_list 的長度相同
        asp_count = modified_text.count('<asp>')
        
        if asp_count == len(category_list):
            # 更新修改過的文本到結果中
            result.append({
                'text': modified_text,
                'category': category_list,
                'category_subcategory': category_subcategory_list
            })
        else:
            print(f"Skipping entry {idx} due to mismatch: <asp> count ({asp_count}) != category count ({len(category_list)})")
    
    return result

preprocessed_data = insert_aspect_tokens(data)

def count_category_subcategory(data):
    category_subcategory_counter = Counter()
    labels_category_subcategory = []
    
    for entry in data:
        categories = entry['category_subcategory']
        category_subcategory_counter.update(categories)
        sorted_categories = category_subcategory_counter.most_common()
    
    # 印出每種category_subcategory的數量
    for category_subcategory, count in sorted_categories:
        print(f"{category_subcategory}: {count}")
        labels_category_subcategory.append(category_subcategory)
    print("Total subcategories:", len(sorted_categories))

    return labels_category_subcategory

def count_category(data):
    category_counter = Counter()
    labels_category = []
    
    for entry in data:
        categories = entry['category']
        category_counter.update(categories)
        sorted_categories = category_counter.most_common()
    
    # 印出每種category_subcategory的數量
    for category, count in sorted_categories:
        print(f"{category}: {count}")
        labels_category.append(category)
    print("Total categories:", len(sorted_categories))

    return labels_category

labels_category = count_category(preprocessed_data)
print("\n")
labels_category_subcategory = count_category_subcategory(preprocessed_data)

train_data, val_data = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-lert-base')
bert_model = BertModel.from_pretrained('hfl/chinese-lert-base')

tokenizer.add_tokens(['<asp>', '</asp>'])

# 更新模型的嵌入層
bert_model.resize_token_embeddings(len(tokenizer))

def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding="max_length", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state
    # 提取 tokenized 之後的文本
    tokenized_text = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    return embeddings, tokenized_text

X_train = []
y_train = []
X_val = []
y_val = []

##### Get LERT embeddings. #####
def process_data(data):
    X = []  # 儲存嵌入
    y_cat = []  # 儲存標籤
    y_cat_subcat = []

    for entry in tqdm(data, desc="Processing Data", unit="entry"):
        input_text = entry['text']
        cat = entry['category']
        cat_subcat = entry['category_subcategory']

        # 獲取文本的 BERT 嵌入
        embeddings, tokenized_text = get_bert_embeddings(input_text)

        # 提取<asp>標記的位置信息
        asp_positions_in_tokens = [i for i, token in enumerate(tokenized_text) if token == '<asp>']
        asp_2_positions_in_tokens = [i for i, token in enumerate(tokenized_text) if token == '</asp>']

        if len(asp_positions_in_tokens) != len(cat):
            print(f"Warning: Number of <asp> tags ({len(asp_positions_in_tokens)}) does not match number of categories ({len(cat)}) for entry: {input_text}")
            print(f"ASP Positions: {asp_positions_in_tokens}, Length: {len(asp_positions_in_tokens)}")
            print("tokenized text: \n", tokenized_text)
            print("text length: ", len(input_text))
            print("tokenized text length: ", len(tokenized_text))
            break

        # 對每個 <asp> 提取對應的嵌入
        asp_embeddings = []
        for pos_asp, pos_asp_2 in zip(asp_positions_in_tokens, asp_2_positions_in_tokens):
            asp_embedding = embeddings[:, pos_asp:pos_asp + 1, :].squeeze()  # <asp> embedding
            asp_2_embedding = embeddings[:, pos_asp_2:pos_asp_2 + 1, :].squeeze()  # </asp> embedding
            
            # 確保兩個嵌入的形狀都是正確的 (768,)
            if asp_embedding.shape == (768,) and asp_2_embedding.shape == (768,):
                # concatenated_embedding = (asp_embedding + asp_2_embedding) / 2
                concatenated_embedding = torch.cat((asp_embedding, asp_2_embedding), dim=-1)  # 將 <asp> 和 </asp> 的嵌入進行串接
                asp_embeddings.append(concatenated_embedding.numpy())# 添加到列表中
            else:
                print(f"Skipping embeddings with shapes: {asp_embedding.shape}, {asp_2_embedding.shape}")

        # 將嵌入和對應的類別添加到列表中
        if asp_embeddings:
            X.extend(asp_embeddings)  # 將每個 <asp> 的嵌入展開
            y_cat.extend(cat)
            y_cat_subcat.extend(cat_subcat)
        
    return np.array(X), y_cat, y_cat_subcat

X_train, y_train_cat_raw, y_train_cat_subcat_raw = process_data(train_data)
X_val, y_val_cat_raw, y_val_cat_subcat_raw = process_data(val_data)

# 將類別標籤轉換為整數
le_cat = LabelEncoder()
y_train_cat_int = le_cat.fit_transform(y_train_cat_raw)
y_val_cat_int = le_cat.transform(y_val_cat_raw)

le_cat_subcat = LabelEncoder()
y_train_cat_subcat_int = le_cat_subcat.fit_transform(y_train_cat_subcat_raw)
y_val_cat_subcat_int = le_cat_subcat.transform(y_val_cat_subcat_raw)

##### Define the MLP model. #####
class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_size, second_hidden_size, num_categories, num_subcategories):
        super(CustomMLP, self).__init__()
        self.dropout = nn.Dropout(0.6) 

        # 第一層共享全連接層
        self.fc1 = nn.Linear(input_size, hidden_size)

        # 第二層共享全連接層
        self.fc2 = nn.Linear(hidden_size, second_hidden_size)
        
        # 分成兩個分支
        # 第一個分支輸出 category
        self.fc_category = nn.Linear(second_hidden_size, num_categories)
        
        # 第二個分支輸出 category_subcategory
        self.fc_subcategory = nn.Linear(second_hidden_size, num_subcategories)
        
        # 激活函數
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 通過共享的第一層
        x = self.relu(self.fc1(x))
        x = self.dropout(x) 

        # 通過第二層共享層
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 分成兩個分支
        category_output = self.fc_category(x)
        subcategory_output = self.fc_subcategory(x)
        
        return category_output, subcategory_output

# 初始化模型
input_size = X_train.shape[1]  # 根據你的輸入特徵數量調整
hidden_size = 128
second_hidden_size = 64
num_categories = len(labels_category)  
num_cat_subcategories = len(labels_category_subcategory)

model = CustomMLP(input_size, hidden_size, second_hidden_size, num_categories, num_cat_subcategories)


weights_category = [3, 3, 1, 5, 1, 1, 1]
weights_subcategory = [1, 5, 5, 1, 1,
                       1, 5, 1, 1, 5,
                       1, 10, 1, 5, 1,
                       10, 1]

# 將權重轉換成 tensor
weights_category_tensor = torch.tensor(weights_category, dtype=torch.float32)
weights_subcategory_tensor = torch.tensor(weights_subcategory, dtype=torch.float32)

# 損失函數和優化器
# 將 class weights 傳入 CrossEntropyLoss 中
criterion_category = nn.CrossEntropyLoss(weight=weights_category_tensor)
criterion_subcategory = nn.CrossEntropyLoss(weight=weights_subcategory_tensor)

optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.02)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

def evaluate(model, X_val, y_val_cat, y_val_subcat):
    model.eval()
    with torch.no_grad():
        category_output, subcategory_output = model(X_val)
        
        # 計算驗證損失
        val_loss_category = criterion_category(category_output, y_val_cat)
        val_loss_subcategory = criterion_subcategory(subcategory_output, y_val_subcat)
        
        val_loss = val_loss_category + val_loss_subcategory
        
        # 將模型的輸出轉換為預測類別
        category_preds = torch.argmax(category_output, dim=1).cpu().numpy()
        subcategory_preds = torch.argmax(subcategory_output, dim=1).cpu().numpy()
        
        # 將標籤轉換為 numpy
        y_val_cat_np = y_val_cat.cpu().numpy()
        y_val_subcat_np = y_val_subcat.cpu().numpy()
        
        # 計算 F1 分數
        f1_category = f1_score(y_val_cat_np, category_preds, average='weighted')
        f1_subcategory_w = f1_score(y_val_subcat_np, subcategory_preds, average='weighted')
        f1_subcategory_m = f1_score(y_val_subcat_np, subcategory_preds, average='macro')
        
        return val_loss.item(), f1_category, f1_subcategory_w, f1_subcategory_m
    
def train(model, X_train, y_train_cat, y_train_subcat, X_val, y_val_cat, y_val_subcat, num_epochs):
    best_f1 = 0.0  # 記錄最佳的 F1 分數
    best_model_state = None  # 用於存儲最佳模型狀態

    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        optimizer.zero_grad()
        
        # 前向傳播
        category_output, subcategory_output = model(X_train)
        
        # 計算損失
        loss_category = criterion_category(category_output, y_train_cat)
        loss_subcategory = criterion_subcategory(subcategory_output, y_train_subcat)
        
        # 加權損失
        loss = 0.3 * loss_category + 1.0 * loss_subcategory
        
        # 反向傳播和優化
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 30 == 0:
            # 計算驗證損失和 F1 分數
            val_loss, f1_cat, f1_subcat_w, f1_subcat_m = evaluate(model, X_val, y_val_cat, y_val_subcat)
            
            # 打印訓練和驗證的損失及 F1 分數
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {loss.item():.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'F1 Category: {f1_cat:.4f}, '
                  f'Weighted F1 Subcategory: {f1_subcat_w:.4f}, '
                  f'Macro F1 Subcategory: {f1_subcat_m:.4f}')
            
            # 調度器調整學習率（根據驗證損失）
            scheduler.step(val_loss)
            
            # 如果 F1 Category 分數比當前最佳值高，則保存模型
            if f1_subcat_w > best_f1:
                best_f1 = f1_subcat_w
                best_model_state = model.state_dict()  # 保存模型狀態
                
                # 保存模型到檔案
                torch.save(best_model_state, 'best_model.pth')
                
    print(f'Best model saved at epoch {epoch + 1} with Weighted F1 Subcategory: {best_f1:.4f}')

# 轉換成 PyTorch 張量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_cat_tensor = torch.tensor(y_train_cat_int, dtype=torch.long)
y_train_subcat_tensor = torch.tensor(y_train_cat_subcat_int, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_cat_tensor = torch.tensor(y_val_cat_int, dtype=torch.long)
y_val_subcat_tensor = torch.tensor(y_val_cat_subcat_int, dtype=torch.long)

#### Train the model. ####
train(model, X_train_tensor, y_train_cat_tensor, y_train_subcat_tensor, 
      X_val_tensor, y_val_cat_tensor, y_val_subcat_tensor, num_epochs=300)


model.eval()
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

with torch.no_grad():
    category_output, subcategory_output = model(X_val_tensor)

# 獲取預測的類別索引
_, predicted_categories = torch.max(category_output, dim=1)  # 主要類別
_, predicted_subcategories = torch.max(subcategory_output, dim=1)  # 子類別

##### Calculate the F1 scores. #####
f1_category = f1_score(y_val_cat_tensor.numpy(), predicted_categories.numpy(), average=None)  
f1_subcategory = f1_score(y_val_subcat_tensor.numpy(), predicted_subcategories.numpy(), average=None)  

print("F1 Scores for Categories:")
for idx, score in enumerate(f1_category):
    print(f"{le_cat.classes_[idx]}: {score:.4f}")

print("\nF1 Scores for Category_Subcategories:")
for idx, score in enumerate(f1_subcategory):
    print(f"{le_cat_subcat.classes_[idx]}: {score:.4f}")