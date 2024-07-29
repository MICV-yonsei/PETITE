from sklearn.model_selection import KFold
import random
import json
import os


data_dir = "/nfs/home/yumin/workspace/CVTGAN/dataset/12812890"

image_path = os.path.join(data_dir, "images")
label_path = os.path.join(data_dir, "labels")

image_file_names = os.listdir(image_path)
label_file_names = os.listdir(label_path)

image_file_names = random.sample(image_file_names, 45)

# KFold
num_files = 45
n = 3

num_test = int(num_files/n)
num_train = num_files - num_test
kf = KFold(n_splits=n)

# Gernerate JSON
for i, (train_index, test_index) in enumerate(kf.split(image_file_names)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
    
    # Convert a JSON file to a dictionary type
    json_file = {
        "name": "adni",
        "tensorImageSize": "3D",
        "training": [],
        "validation": [],
    }
    # Train
    for j in range(num_train):
        image_name = image_file_names[train_index[j]]
        image_num = image_name.split("_")[3]
        try: 
            label_name = [label_name for label_name in label_file_names if label_name.split("_")[3]==image_num][0]
            if image_name not in json_file:
                json_file["training"].append({
                    "image": "images/" + image_name,
                    "label": "labels/" + label_name
                })
        except:
            pass
        
    # Test (Validation)
    for k in range(num_test):
        image_name = image_file_names[test_index[k]]
        image_num = image_name.split("_")[3]
        try: 
            label_name = [label_name for label_name in label_file_names if label_name.split("_")[3]==image_num][0]
            if image_name not in json_file:
                json_file["validation"].append({
                    "image": "images/" + image_name,
                    "label": "labels/" + label_name
                })
        except:
            pass
            
    print("* Finish generated JSON File- Num_training files:",len(json_file["training"]), "Num_validation files",len(json_file["validation"]))
    json_file_name = os.path.join(data_dir, f"Resolution{i}.json")
    with open(json_file_name, "w", encoding="utf-8") as file:
        json.dump(json_file, file, indent="\t")