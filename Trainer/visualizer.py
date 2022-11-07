import argparse
from utils.utils import read_cfg
from data import covid_dataloader
import torch
import matplotlib.pyplot as plt
import cv2

INDEX_TO_LABEL = {
    0: "COVID",
    1: "Lung_Opacity",
    2: "Normal",
    3: "Viral Pneumonia"
}

def iterate_and_save_augm_img(sub_dataset, data_name):
    
    imgs_list = torch.Tensor()
    labels_list = list()
    for i, (imgs, labels) in enumerate(sub_dataset):
        imgs_list = imgs
        labels_list = labels.tolist()
        break
    
    for i, (img, label) in enumerate(zip(imgs_list, labels_list)):
        img = img.permute(1, 2, 0).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(label)
        plt.imshow(img)
        plt.savefig(f"Sample-{INDEX_TO_LABEL[label]}-{data_name}-{i}.png")




def visualize_dataset(dataset: covid_dataloader.CovidDataModule):
    list_dataloader = [dataset.train_dataloader() ,dataset.val_dataloader() ,dataset.test_dataloader()]
    dataset_name = ["train", "val", "test"]

    for name, dl in zip(dataset_name, list_dataloader):
        iterate_and_save_augm_img(dl, name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument for train the model")
    parser.add_argument('-cfg', '--config', type=str, help="Path to config yaml file")
    # parser.add_argument() # TODO: add params max imgs
    # parser.add_argument() # TODO: add params dest folder.
    args = parser.parse_args()
    cfg = read_cfg(cfg_file=args.config)
    dataset = covid_dataloader.CovidDataModule(cfg)
    visualize_dataset(dataset)