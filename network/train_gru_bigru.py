from tqdm import tqdm
from DNNs import FCNet, GRUNet
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from Data_utils import DescriptionSceneDataset, DescriptionSceneDatasetCombined
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import train_utility


def contrastive_loss(pairwise_distances, targets, margin=0.25):
    batch_size = pairwise_distances.shape[0]
    diag = pairwise_distances.diag().view(batch_size, 1)
    pos_masks = torch.eye(batch_size).bool().to(pairwise_distances.device)
    d1 = diag.expand_as(pairwise_distances)
    cost_s = (margin + pairwise_distances - d1).clamp(min=0)
    cost_s = cost_s.masked_fill(pos_masks, 0)
    cost_s = cost_s / (batch_size * (batch_size - 1))
    cost_s = cost_s.sum()

    d2 = diag.t().expand_as(pairwise_distances)
    cost_d = (margin + pairwise_distances - d2).clamp(min=0)
    cost_d = cost_d.masked_fill(pos_masks, 0)
    cost_d = cost_d / (batch_size * (batch_size - 1))
    cost_d = cost_d.sum()

    return (cost_s + cost_d) / 2


def collate_fn(data):
    # desc
    tmp_data = [x[0] for x in data]
    tmp = pad_sequence(tmp_data, batch_first=True)
    desc_ = pack_padded_sequence(tmp,
                                 torch.tensor([len(x) for x in tmp_data]),
                                 batch_first=True,
                                 enforce_sorted=False)
    # scenes
    tmp_scene = [x[1] for x in data]
    scenes_ = torch.stack(tmp_scene)

    return desc_, scenes_


def start_train():
    type_room = 'bedroom'
    print("type_room:", type_room)

    data_description_living = data_description_bedroom = data_scene_living = data_scene_bedroom = None
    data_description_ = data_scene_ = None

    no_of_videos = 25
    out_put_feature_size = 256

    is_bidirectional = True
    model_descriptor = GRUNet(hidden_size=out_put_feature_size, num_features=768, is_bidirectional=is_bidirectional)
    model_scene = FCNet(input_size=400, feature_size=out_put_feature_size)

    type_model_desc = 'bigru' if is_bidirectional else 'gru'

    num_epochs = 50
    batch_size = 64

    # Loading Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model_descriptor.to(device)
    model_scene.to(device)

    # Loading Data
    if type_room in ['living', 'bedroom']:
        data_description_, data_scene_ = train_utility.get_entire_data(type_room=type_room, no_of_videos=no_of_videos)
        train_indices, val_indices, test_indices = train_utility.retrieve_indices(data_scene_.size()[0], type_room)
        dataset = DescriptionSceneDataset(data_description_, data_scene_, type_model_desc="gru")

    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    params = list(model_descriptor.parameters()) + list(model_scene.parameters())
    optimizer = torch.optim.Adam(params, lr=0.008)

    train_losses = []
    val_losses = []

    # Define the StepLR scheduler
    step_size = 17
    gamma = 0.75
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    if is_bidirectional:
        file_prefix = "bidirectional_gru_" + type_room + '_' + str(no_of_videos)
    else:
        file_prefix = "gru_" + type_room + '_' + str(no_of_videos)

    r10_hist = []
    best_r10 = 0

    for epoch in tqdm(range(num_epochs)):
        total_loss_train = 0
        total_loss_val = 0
        num_batches_train = 0
        num_batches_val = 0
        for i, (data_description, data_scene) in enumerate(train_loader):
            data_scene = data_scene.to(device)
            data_description = data_description.to(device)

            # zero the gradients
            optimizer.zero_grad()

            output_descriptor = model_descriptor(data_description)
            output_scene = model_scene(data_scene)

            # transposed_scene_features = torch.transpose(output_scene, 0, 1)
            # multiplication = torch.mm(output_descriptor, transposed_scene_features)
            multiplication = train_utility.cosine_sim(output_descriptor, output_scene)

            # print(multiplication.diagonal().mean().item())
            # define the target
            ground_truth = torch.eye(multiplication.size()[0], device=device)

            loss = contrastive_loss(multiplication, targets=ground_truth)

            loss.backward()

            optimizer.step()

            total_loss_train += loss.item()
            num_batches_train += 1

        scheduler.step()
        print(scheduler.get_last_lr())
        epoch_loss_train = total_loss_train / num_batches_train

        # Evaluate validation sets
        with torch.no_grad():
            for j, (data_description, data_scene) in enumerate(val_loader):
                data_description = data_description.to(device)
                data_scene = data_scene.to(device)
                output_descriptor = model_descriptor(data_description)
                output_scene = model_scene(data_scene)

                multiplication = train_utility.cosine_sim(output_descriptor, output_scene)

                # define the target
                ground_truth = torch.eye(multiplication.size()[0])
                ground_truth = ground_truth.to(device)

                loss = contrastive_loss(multiplication, targets=ground_truth)

                total_loss_val += loss.item()

                num_batches_val += 1

            epoch_loss_val = total_loss_val / num_batches_val

        r1, r5, r10, _, _, _, _, _, _, _ = train_utility.evaluate_model(model_descriptor=model_descriptor,
                                                                        model_scene=model_scene,
                                                                        data_description=data_description_,
                                                                        data_scene=data_scene_,
                                                                        data_description_living=data_description_living,
                                                                        data_description_bedroom=data_description_bedroom,
                                                                        data_scene_living=data_scene_living,
                                                                        data_scene_bedroom=data_scene_bedroom,
                                                                        section="val",
                                                                        no_vids=no_of_videos,
                                                                        indices=val_indices,
                                                                        type_model_desc=type_model_desc,
                                                                        type_room=type_room)
        r10_hist.append(r10)
        if r10 > best_r10:
            best_r10 = r10
            train_utility.save_best_model(model_scene.state_dict(), model_descriptor.state_dict(), file_prefix + '.pt')

        print("train_loss:", epoch_loss_train)
        print("val_loss:", epoch_loss_val)

        train_losses.append(epoch_loss_train)
        val_losses.append(epoch_loss_val)

    # load best model for the evaluation stage
    best_model_state_dict_scene, best_model_state_dict_description = train_utility.load_best_model(file_prefix + '.pt')
    model_scene.load_state_dict(best_model_state_dict_scene)
    model_descriptor.load_state_dict(best_model_state_dict_description)

    r1, r5, r10, ndgc_10, ndcg = train_utility.evaluate_video_queries(model_descriptor=model_descriptor,
                                                                      model_scene=model_scene,
                                                                      data_scene=data_scene_,
                                                                      indices=test_indices,
                                                                      batch_size=batch_size,
                                                                      no_of_videos=no_of_videos,
                                                                      type_room=type_room,
                                                                      data_scene_living=data_scene_living,
                                                                      data_scene_bedroom=data_scene_bedroom,
                                                                      desc_model_type='gru')

    r1, r5, r10, ndgc_10, ndcg = train_utility.evaluate_distance_queries(model_descriptor=model_descriptor,
                                                                         model_scene=model_scene,
                                                                         data_scene=data_scene_,
                                                                         indices=test_indices,
                                                                         type_room=type_room,
                                                                         desc_model_type='gru',
                                                                         batch_size=batch_size,
                                                                         data_scene_living=data_scene_living,
                                                                         data_scene_bedroom=data_scene_bedroom)

    r1, r5, r10, ndgc_10, ndcg = train_utility.evaluate_style_queries(model_descriptor=model_descriptor,
                                                                      model_scene=model_scene,
                                                                      data_scene=data_scene_,
                                                                      indices=test_indices,
                                                                      desc_model_type='gru',
                                                                      type_room=type_room,
                                                                      data_scene_living=data_scene_living,
                                                                      data_scene_bedroom=data_scene_bedroom,
                                                                      batch_size=batch_size)

    ds1, ds5, ds10, sd1, sd5, sd10, ndgc_10, ndcg, ds_medr, sd_medr = train_utility.evaluate_model(model_descriptor,
                                                                                                   model_scene,
                                                                                                   data_description_,
                                                                                                   data_scene_,
                                                                                                   data_description_living=data_description_living,
                                                                                                   data_description_bedroom=data_description_bedroom,
                                                                                                   data_scene_living=data_scene_living,
                                                                                                   data_scene_bedroom=data_scene_bedroom,
                                                                                                   section="test",
                                                                                                   no_vids=no_of_videos,
                                                                                                   indices=test_indices,
                                                                                                   type_model_desc=type_model_desc,
                                                                                                   type_room=type_room)
    train_utility.write_models_evaluation_to_file([ds1, ds5, ds10, sd1, sd5, sd10, ndgc_10, ndcg, ds_medr, sd_medr],
                                                  file_prefix + "_test_evaluate" + ".txt")

    print("best_r10", best_r10)


if __name__ == '__main__':
    start_train()
