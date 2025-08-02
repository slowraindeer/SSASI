#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torchvision.models as models
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


# In[2]:


dir = '/data/nas2/KJA/MIS/LIDC dataset/'


# A = [dir+'/Consistency/BB/A/'+items for items in os.listdir(dir+'/Consistency/BB/A/')]
# B = [dir+'/Consistency/BB/B/'+items for items in os.listdir(dir+'/Consistency/BB/B/')]
# A.sort()
# B.sort()
# B = B[1:601]
# A = A[1:]

# In[3]:


A_img = [dir+'/A/'+items for items in os.listdir(dir+'A')]
B_img = [dir+'/B/'+items for items in os.listdir(dir+'B')]
A_img.sort()
B_img.sort()
A_img = A_img[:125]
B_img = B_img[:125]


# In[4]:


B_labels = np.zeros((125)).astype(int)
A_labels = np.ones((125)).astype(int)


# In[5]:


img_pairs = []
img_pair_labels = []

for i in range(125):
    currentA = A_img[i]
    pos_img = A_img[124-i]
    img_pairs.append([currentA, pos_img])
    img_pair_labels.append(1)
for i in range(125):
    currentA = A_img[i]
    pos_B = B_img[124-i]
    img_pairs.append([currentA, pos_B])
    img_pair_labels.append(0)


# In[6]:


class AnnotationStyleDataset(Dataset):
    def __init__(self, image_pairs, labels, transform=None):
        # self.mask_pairs = mask_pairs
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform


    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        # mask1_path, mask2_path = self.mask_pairs[idx]
        img1_path, img2_path = self.image_pairs[idx]

        # mask1 = Image.open(mask1_path).convert('L')
        # mask2 = Image.open(mask2_path).convert('L')

        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        label = self.labels[idx]

        # mask1_np = np.array(mask1)  # Shape: [H, W]
        # mask2_np = np.array(mask2)
        img1_np = np.array(img1)
        img2_np = np.array(img2)

        fin1_np = np.stack([img1_np, img1_np, img1_np], axis=-1)  # Shape: [H, W, 3]
        fin2_np = np.stack([img2_np, img2_np, img2_np], axis=-1)  # Shape: [H, W, 3]

        # Convert back to PIL Image (so that transforms can be applied)
        fin1 = Image.fromarray(fin1_np)
        fin2 = Image.fromarray(fin2_np)

        if self.transform:
            fin1 = self.transform(fin1)
            fin2 = self.transform(fin2)

        return fin1, fin2, torch.tensor(label, dtype=torch.float32)


# In[7]:


import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = models.densenet161(pretrained=True)
        self.cnn.classifier = nn.Identity()  # Remove classification head

        self.fc1 = nn.Linear(2210, 1024)  # 2208 features + 2 Hu moments
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)

    def extract_hu_moments(self, img):
        binary = img[:][0] > 1
        binary = binary.astype(np.uint8)
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        hu_selected = np.array([hu_moments[2], hu_moments[6]])
        return hu_selected

    def forward_once(self, x):
        batch_size = x.shape[0]
        x_np = x.permute(0, 2, 3, 1).cpu().numpy()
        hu_moments_batch = [self.extract_hu_moments(img) for img in x_np]
        hu_moments_tensor = torch.tensor(hu_moments_batch, dtype=torch.float32, device=x.device)

        x = self.cnn.features(x)  # (B, 2208, 7, 7)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)  # (B, 2208)
        x = torch.cat((x, hu_moments_tensor), dim=1)  # (B, 2210)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.dropout(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# In[8]:


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((label * torch.pow(euclidean_distance, 2) +
                          (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)))
        return loss


# In[9]:


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = AnnotationStyleDataset(img_pairs, img_pair_labels, transform=transform)

train_idx, val_idx = train_test_split(
    range(len(dataset)),  # Generates an index list from 0 to len(dataset)-1
    test_size=0.3,  # 30% of the data goes into the validation set
    stratify=img_pair_labels,  # Stratify the split based on the labels
    random_state=18  # For reproducibility
)

# Create subsets for training and validation
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=16)


# In[10]:


model = SiameseNetwork()
device = torch.device("cuda:0")
model = model.to(device)

criterion = ContrastiveLoss(margin=1)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# In[11]:


best_val_loss = float('inf')
patience = 50  # Number of epochs to wait before stopping
counter = 0

num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    num_train_samples = 0

    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
        for img1, img2, label in tepoch:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device).float()

            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * img1.size(0)
            num_train_samples += img1.size(0)
            tepoch.set_postfix(loss=loss.item())

    avg_train_loss = total_train_loss / num_train_samples

    model.eval()
    total_val_loss = 0.0
    num_val_samples = 0

    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device).float()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            total_val_loss += loss.item() * img1.size(0)
            num_val_samples += img1.size(0)

    avg_val_loss = total_val_loss / num_val_samples
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(model.state_dict(), 'Siamese_AAAB_LIDC1.pth')
        print("✅ Best model saved!")
    else:
        counter += 1
        print(f"⏳ Early stopping counter: {counter}/{patience}")

    if counter >= patience:
        print("⛔ Early stopping triggered! Training stopped.")
        break


# In[12]:


model.load_state_dict(torch.load('Siamese_AAAB_LIDC1.pth'))
model. to(device)


# In[13]:


def collect_test(img_dir, label):
    file_names = sorted(f for f in os.listdir(img_dir) if f != '@eaDir')
    img_paths = [os.path.join(img_dir, f.replace('.png', '.png')) for f in file_names]
    return list(zip(img_paths, [label] * len(file_names)))


# In[14]:


test_dataset = []

test_dataset += collect_test(
    dir + "/A_test/",
    1
)

test_dataset += collect_test(
    dir + "/B_test/",
    0
)


# In[15]:


test_dataset = test_dataset[:3187]+test_dataset[-3188:]


# In[16]:


len(test_dataset)


# In[17]:


def build_support_tensor(img_path):
    img = Image.open(img_path).convert('L')     # [1, H, W] # still [1, H, W]
    img = np.array(img)
    combined = np.stack([img, img, img], axis=-1)     # [3, H, W]

    combined = Image.fromarray(combined)
    combined = transform(combined)
    return combined


# In[18]:


img_dir = os.path.join('/data/nas2/KJA/MIS/', 'LIDC dataset/A/')

A_images_list = []

for mask_filename in sorted(os.listdir(img_dir)):
    if not mask_filename.endswith('.png'):
        pass
    base_name = os.path.splitext(mask_filename)[0]  # remove .png
    img_filename = base_name + '.png'
    img_path = os.path.join(img_dir, img_filename)

    if os.path.exists(img_path):
        A_images_list.append(img_path)
    else:
        print(f"❗ Warning: No matching image found for {mask_filename}")

# Build A_test tensors
A_refs = []
for A_img in A_images_list[:50]:
    A_ref = build_support_tensor(A_img).unsqueeze(0)
    A_refs.append(A_ref)
A_refs = [ref.to(device) for ref in A_refs]


# In[19]:


from tqdm import tqdm

def evaluate_with_auroc(model, test_data, device):
    model.eval()
    all_labels = []
    all_distances = []

    # === Step 1: Preload and stack A_refs into a batch ===
    A_ref_batch = torch.stack([ref.to(device) for ref in A_refs])  # Shape: (N_refs, C, H, W)
    num_refs = A_ref_batch.size(0)
    if A_ref_batch.dim() == 5 and A_ref_batch.shape[1]==1:
        A_ref_batch = A_ref_batch.squeeze(1)


    with torch.no_grad():
        for img_path, label in tqdm(test_data, desc="Evaluating AUROC"):
            # Step 2: Load and repeat query
            query_tensor = build_support_tensor(img_path).unsqueeze(0).to(device)
            query_batch = query_tensor.repeat(num_refs, 1, 1, 1)  # Shape: (N_refs, C, H, W)

            if query_batch.dim() == 5 and query_batch.shape[1] == 1:
                query_batch = query_batch.squeeze(1)

            # Step 3: Batched inference
            output1, output2 = model(A_ref_batch, query_batch)
            squared_distances = torch.pow(F.pairwise_distance(output1, output2), 2)  # Shape: (N_refs,)

            # Step 4: Average distance
            mean_distance = squared_distances.mean().item()

            all_distances.append(mean_distance)
            all_labels.append(label)

    # === Compute AUROC ===
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    # Negate distances: smaller distance = more similar → higher score
    fpr, tpr, thresholds = roc_curve(all_labels, -all_distances)
    roc_auc = auc(fpr, tpr)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Choose a threshold (e.g., one that gives best TPR/FPR balance)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Convert distances to predicted labels (negated because smaller = more similar)
    y_pred = (-all_distances > optimal_threshold).astype(int)


    # Confusion matrix
    cm = confusion_matrix(all_labels, y_pred)

    tn, fp, fn, tp = cm.ravel()

# Metrics
    accuracy = accuracy_score(all_labels, y_pred)
    precision = precision_score(all_labels, y_pred)
    recall = recall_score(all_labels, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(all_labels, y_pred)

    # Prediction distribution
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_distribution = dict(zip(unique, counts))

    # Display
    print(f"Best threshold from ROC: {optimal_threshold:.4f}")
    print(f"Min distance: {np.min(all_distances):.4f}, Max distance: {np.max(all_distances):.4f}")
    print(f"Prediction distribution: {pred_distribution}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Sensitivity (Recall): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Optional: Class names
    class_names = ['0', '1']

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix @ Threshold={optimal_threshold:.4f}')
    plt.tight_layout()
    plt.show()


    # === Plot ROC ===
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return fpr, tpr, thresholds, roc_auc


# In[20]:


fpr, tpr, thresholds, roc_auc = evaluate_with_auroc(model, test_dataset, device)


# In[ ]:




