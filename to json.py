import json
import boto3
import os

bucket_name = "minseok-mask-detection-train-dataset"
with_mask_dir = "mask_images/with_mask"
without_mask_dir = "mask_images/without_mask"
augmented_postfix = "Augmented"
metadata_dir = "mask_images/metadata"

s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket(bucket_name) 
with_mask_images_list_in_dir = list(bucket.objects.filter(Prefix = with_mask_dir))
without_mask_images_list_in_dir = list(bucket.objects.filter(Prefix = without_mask_dir))

json_datas = []
for img_info in with_mask_images_list_in_dir:
    json_data = {}
    file_path = img_info.key

    json_data["class"] = "with_mask"
    file_name = file_path.split("/")[-1]
    json_data["file_name"] = file_name
    if file_name.split("_")[0] == "Augmented":
        json_data["is_augmented"] = True
    else:
        json_data["is_augmented"] = False
    json_datas.append(json_data)
        
for img_info in without_mask_images_list_in_dir:
    json_data = {}
    file_path = img_info.key

    json_data["class"] = "without_mask"
    file_name = file_path.split("/")[-1]
    json_data["file_name"] = file_name
    if file_name.split("_")[0] == "Augmented":
        json_data["is_augmented"] = True
    else:
        json_data["is_augmented"] = False
    json_datas.append(json_data)

if __name__ == "__main__":
    os.makedirs("metadata", exist_ok=True)
    for json_data in json_datas:
        
        with open(f'metadata/{json_data["file_name"]}.json', 'w') as file:
            json.dump(json_data, file)
