import torch

'''
This module aims to concat the embedding of each scenario with the corresponding multimedia of it.
'''

no_using_video_captions = 25


def start_merge():
    _type = '_living'
    device = "cpu"
    data_scene_ = torch.load(f'./scene_features/final_tensor_scenes_200_including_tv{_type}.pt')
    data_video_ = torch.load('./data_youcook2/representations.pt')
    data_scene_ = data_scene_.to(device)
    data_video_ = data_video_.to(device)
    scene_video_data = torch.empty(data_scene_.shape[0], data_scene_.shape[1] + data_video_.shape[1])
    for i in range(data_scene_.shape[0]):
        scene_video_data[i, :] = torch.cat((data_scene_[i], data_video_[i % no_using_video_captions]), dim=0)
    torch.save(torch.tensor(scene_video_data), './scene_features/scene_video_data_'+str(no_using_video_captions)+_type+'.pt')


if __name__ == '__main__':
    start_merge()
