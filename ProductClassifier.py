import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import os

# Load CSV and drop bad rows
df = pd.read_csv('styles.csv', on_bad_lines='skip')
df = df.dropna(subset=['gender', 'masterCategory', 'subCategory', 'articleType'])

# Create image name column
image_dir = 'images/'
df['image_name'] = df['id'].astype(str) + '.jpg'

# Keep only rows for which images actually exist
df = df[df['image_name'].isin(os.listdir(image_dir))]

# Add desired target classes (including Heels & Sandals)
target_classes = ['Tshirts', 'Dresses', 'Jeans', 'Shorts', 'Heels', 'Sandals']
df = df[df['articleType'].isin(target_classes)]

# Create train/test folders for each class
train_dir = 'dataset/train'
test_dir = 'dataset/test'

for c in target_classes:
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(test_dir, c), exist_ok=True)

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['articleType'], random_state=42)

# Copy images to respective folders
def move_images(data, base_dir):
    for _, row in data.iterrows():
        fname = row['image_name']
        label = row['articleType']
        src = os.path.join(image_dir, fname)
        dst = os.path.join(base_dir, label, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)

move_images(train_df, train_dir)
move_images(test_df, test_dir)

print("✅ Images copied into subfolders for all selected classes.")
