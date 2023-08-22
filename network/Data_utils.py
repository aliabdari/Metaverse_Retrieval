import os
import random
import torch
from torch.utils.data import Dataset
import pickle
import time


# Scene and Description dataset
class DatasetMean(Dataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        x1 = self.data1[index]
        x2 = self.data2[index]
        return x1, x2


class DescriptionSceneDataset(Dataset):
    def __init__(self, data_description_path, data_scene, type_model_desc):
        self.description_path = data_description_path
        self.data_scene = data_scene
        self.type_model_desc = type_model_desc

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        data_description = torch.load(self.description_path + os.sep + "desc_" + str(index) + ".pt")
        if self.type_model_desc == "mean":
            data_description = torch.mean(data_description, 0)
        data_scene = self.data_scene[index]
        return data_description, data_scene


class DescriptionSceneDatasetCombined(Dataset):
    def __init__(self, data_description_living, data_description_bedroom, data_scene_living, data_scene_bedroom,
                 type_model_desc):
        self.description_path_living = data_description_living
        self.description_path_bedroom = data_description_bedroom
        self.data_scene_living = data_scene_living
        self.data_scene_bedroom = data_scene_bedroom
        self.type_model_desc = type_model_desc

        with open('indices/indices_combined_guidance.pkl', 'rb') as f:
            self.indices_guidance = pickle.load(f)

    def __len__(self):
        return len(self.data_scene_living) + len(self.data_scene_bedroom)

    def __getitem__(self, index):
        retrieve_orig = self.indices_guidance[index]
        if retrieve_orig['type'] == 'living':
            data_description = torch.load(
                self.description_path_living + os.sep + "desc_" + str(retrieve_orig['index']) + ".pt")
            data_scene = self.data_scene_living[retrieve_orig['index']]
        elif retrieve_orig['type'] == 'bedroom':
            data_description = torch.load(
                self.description_path_bedroom + os.sep + "desc_" + str(retrieve_orig['index']) + ".pt")
            data_scene = self.data_scene_bedroom[retrieve_orig['index']]
        else:
            raise Exception("Error the type of the room is not recognized")
        if self.type_model_desc == "mean":
            data_description = torch.mean(data_description, 0)
        return data_description, data_scene


class DescriptionSceneDatasetHouseMean(Dataset):
    def __init__(self, data_description_path, data_scene_living, data_scene_bedroom, no_house, type_model_scene):
        self.description_path = data_description_path
        self.data_scene_living = data_scene_living
        self.data_scene_bedroom = data_scene_bedroom
        self.no_houses = no_house
        self.type_model_scene = type_model_scene

        root_path = '/media/HDD/aabdari/PycharmProjects/Sync2Gen-main/dataset_3dfront/outputs_house'
        houses_data = open(root_path + f'/created_house_dataset_indexes_{self.no_houses}.pkl', 'rb')
        self.houses_data = pickle.load(houses_data)

    def __len__(self):
        return len(self.houses_data)

    def __getitem__(self, index):
        data_description = torch.load(self.description_path + os.sep + "desc_" + str(index.item()) + ".pt")

        retrieve_orig = self.houses_data[index]
        livingroom_idx = retrieve_orig['living_room']
        bedrooms_idx = retrieve_orig['bedroom']

        data_scene_ = torch.zeros(1 + len(bedrooms_idx), self.data_scene_bedroom.size()[1])
        data_scene_[0, :] = self.data_scene_living[livingroom_idx]
        for i, j in enumerate(bedrooms_idx):
            data_scene_[i + 1, :] = self.data_scene_bedroom[j]
        if self.type_model_scene == 'mean':
            data_scene_ = torch.mean(data_scene_, 0)

        return data_description, data_scene_


class DatasetsRecVersionRandom(Dataset):
    def __init__(self, data_description_path, data_scene):
        self.description_path = data_description_path
        self.data_scene = data_scene

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        x1 = torch.load(self.description_path + os.sep + "desc_" + str(index.item()) + ".pt")
        no_of_sentences = random.randrange(1, x1.size()[0] - 1)
        selected_list = random.sample(range(0, x1.size()[0] - 1), no_of_sentences)
        x1_new = x1[selected_list]
        x2 = self.data_scene[index]
        return x1_new, x2


class DatasetsRecVersionV2(Dataset):
    def __init__(self, data_scene):
        self.data_scene = data_scene

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        print(index.item())
        x = self.data_scene[index]
        return x


class DatasetsRecIncludingVideo(Dataset):
    def __init__(self, data_description_path, data_scene, representation_video):
        self.description_path = data_description_path
        self.data_scene = data_scene
        self.representation_video = representation_video

    def __len__(self):
        return len(self.data_scene)

    def __getitem__(self, index):
        x1 = torch.load(self.description_path + os.sep + "desc_" + str(index.item()) + ".pt")
        x2_1 = self.data_scene[index]
        x2_2 = self.representation_video[index % 500][:]
        x2 = torch.cat((x2_1, x2_2), dim=0)
        return x1, x2
