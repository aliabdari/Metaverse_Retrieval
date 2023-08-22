import os
import pickle
import math
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from transformers import BertTokenizer, BertModel
import spacy
from textblob import Word
import matplotlib.pyplot as plt


def get_entire_data(type_room, no_of_videos):
    if type_room in ['living', 'bedroom']:
        data_description_ = '../descriptions/sentence_features/description_tensors_vid' + str(no_of_videos) + '_' + type_room
        data_scene_ = torch.load('../scene_features/scene_video_data_' + str(no_of_videos) + '_' + type_room + '.pt')
        return data_description_, data_scene_


def get_scene_data(data_scene_living, data_scene_bedroom, indices):
    with open('./indices/indices_combined_guidance.pkl', 'rb') as f:
        indices_guidance = pickle.load(f)
    if indices is not None:
        data_scene = torch.empty(len(indices), data_scene_bedroom.size()[1])
        index = 0
        for i in indices:
            retrieve_orig = indices_guidance[i]
            if retrieve_orig['type'] == 'living':
                data_scene[index, :] = data_scene_living[retrieve_orig['index']]
            elif retrieve_orig['type'] == 'bedroom':
                data_scene[index, :] = data_scene_bedroom[retrieve_orig['index']]
            index += 1
    else:
        data_scene = torch.cat((data_scene_living, data_scene_bedroom), 0)

    return data_scene


def retrieve_indices(data_size, type_room):
    if type_room == 'bedroom':
        if os.path.isfile('./indices/indices_bedroom.pkl'):
            indices_pickle = open('indices/indices_bedroom.pkl', "rb")
            indices_pickle = pickle.load(indices_pickle)
            train_indices = indices_pickle["train"]
            val_indices = indices_pickle["val"]
            test_indices = indices_pickle["test"]
        else:
            train_ratio = .7
            val_ratio = .15
            perm = torch.randperm(data_size)
            train_indices = perm[:int(data_size * train_ratio)]
            val_indices = perm[int(data_size * train_ratio):int(data_size * (val_ratio + train_ratio))]
            test_indices = perm[int(data_size * (val_ratio + train_ratio)):]
            indices_pickle = {"train": train_indices, "val": val_indices, "test": test_indices}
            with open('./indices/indices_bedroom.pkl', 'wb') as f:
                pickle.dump(indices_pickle, f)
    elif type_room == 'living':
        if os.path.isfile('./indices/indices_living.pkl'):
            indices_pickle = open('./indices/indices_living.pkl', "rb")
            indices_pickle = pickle.load(indices_pickle)
            train_indices = indices_pickle["train"]
            val_indices = indices_pickle["val"]
            test_indices = indices_pickle["test"]
        else:
            train_ratio = .7
            val_ratio = .15
            perm = torch.randperm(data_size)
            train_indices = perm[:int(data_size * train_ratio)]
            val_indices = perm[int(data_size * train_ratio):int(data_size * (val_ratio + train_ratio))]
            test_indices = perm[int(data_size * (val_ratio + train_ratio)):]
            indices_pickle = {"train": train_indices, "val": val_indices, "test": test_indices}
            with open('./indices/indices_living.pkl', 'wb') as f:
                pickle.dump(indices_pickle, f)
    else:
        raise Exception("The type of room is not recognized")

    return train_indices, val_indices, test_indices


def create_percent_queries(scene_result, entire_descriptor, desired_output_indexes):
    similarity = torch.nn.functional.cosine_similarity(entire_descriptor, scene_result, dim=1)
    similarity = similarity.squeeze()
    sorted_indices = torch.argsort(similarity, descending=True)
    sorted_indices = sorted_indices.tolist()
    r_50 = 0
    r_10 = 0
    r_5 = 0
    r_1 = 0

    for j in sorted_indices[:50]:
        if j in desired_output_indexes:
            r_50 = 1

    for j in sorted_indices[:10]:
        if j in desired_output_indexes:
            r_10 = 1

    for j in sorted_indices[:5]:
        if j in desired_output_indexes:
            r_5 = 1

    for j in sorted_indices[:1]:
        if j in desired_output_indexes:
            r_1 = 1

    return r_50, r_10, r_5, r_1


def dcg_idcg_calculator(n, desired_output_indexes, sorted_indices):
    dcg = 0
    idcg = 0
    if n == "entire":
        length_desired_list = len(desired_output_indexes)
        for i, j in enumerate(sorted_indices[:length_desired_list]):
            idcg += 1 / math.log2(i + 2)
            if j in desired_output_indexes:
                dcg += 1 / math.log2(i + 2)
    else:
        length_desired_list = min(len(desired_output_indexes), n)
        for i, j in enumerate(sorted_indices[:length_desired_list]):
            idcg += 1 / math.log2(i + 2)
            if j in desired_output_indexes:
                dcg += 1 / math.log2(i + 2)
    return dcg / idcg


def get_desired_relevant(selected_desc_index, no_of_videos, section, type_room):
    with open('./relevances/relevance_'
              + str(no_of_videos)
              + '_' + section
              + '_' + type_room + '.pkl', 'rb') as f:
        info = pickle.load(f)
    return info[selected_desc_index]


def save_best_model(best_model_state_dict_scene, best_model_state_dict_description, model_name):
    model_path = "models"
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_path = model_path + os.sep + model_name
    torch.save({'model_state_dict_scene': best_model_state_dict_scene,
                'model_state_dict_description': best_model_state_dict_description},
               model_path)


def load_best_model(model_name):
    model_path = "models"
    model_path = model_path + os.sep + model_name
    check_point = torch.load(model_path)
    best_model_state_dict_scene = check_point['model_state_dict_scene']
    best_model_state_dict_description = check_point['model_state_dict_description']
    return best_model_state_dict_scene, best_model_state_dict_description


def write_train_history_to_file(data, file_name):
    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + os.sep + file_name
    if os.path.exists(path):
        os.remove(path)
    with open(path, 'w') as f:
        for d in data:
            f.write(str(d))
            f.write('\n')


def write_models_evaluation_to_file(data, file_name):
    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + os.sep + file_name
    if os.path.exists(path):
        os.remove(path)
    tags = ["ds1", "ds5", "ds10", "sd1", "sd5", "sd10", "ndcg_10", "ndcg", "median_rank_ds", "median_rank_sd"]
    with open(path, 'w') as f:
        for i, d in enumerate(data):
            f.write(tags[i] + " : " + str(d))
            f.write('\n')


def write_queries_eval_to_file(data, file_name):
    path = './results'
    if not os.path.exists(path):
        os.mkdir(path)
    path = path + os.sep + file_name
    if os.path.exists(path):
        os.remove(path)
    tags = ["r1", "r5", "r10", "ndcg_10", "ndcg"]
    with open(path, 'w') as f:
        for i, d in enumerate(data):
            f.write(tags[i] + " : " + str(d))
            f.write('\n')


def create_rank(result, entire_descriptor, desired_output_index):
    similarity = torch.nn.functional.cosine_similarity(entire_descriptor, result, dim=1)
    similarity = similarity.squeeze()
    sorted_indices = torch.argsort(similarity, descending=True)
    position = torch.where(sorted_indices == desired_output_index)
    return position[0].item(), sorted_indices


def get_related_descriptions(data_description_path, indices, type_model_desc):
    tensor_lists = []
    if indices is not None:
        if type_model_desc == 'mean':
            data_description = torch.empty(len(indices), 768)
            index = 0
            for idx in indices:
                data_description[index, :] = torch.mean((torch.load(data_description_path + os.sep + "desc_" + str(idx.item()) + ".pt")), 0)
                index += 1
            return data_description
        else:
            for idx in indices:
                tensor_lists.append(torch.load(data_description_path + os.sep + "desc_" + str(idx.item()) + ".pt"))
    else:
        number_of_files = 0
        for file in os.listdir(data_description_path):
            if file.endswith(".pt"):
                number_of_files += 1
        if type_model_desc == 'mean':
            data_description = torch.empty(number_of_files, 768)
            index = 0
            for idx in range(number_of_files):
                data_description[index, :] = torch.mean((torch.load(data_description_path + os.sep + "desc_" + str(idx) + ".pt")), 0)
                index += 1
            return data_description
        else:
            for idx in range(number_of_files):
                tensor_lists.append(torch.load(data_description_path + os.sep + "desc_" + str(idx) + ".pt"))
    return tensor_lists


def get_list_tensor_description(data_description_path, indices):

    tensor_lists = get_related_descriptions(data_description_path=data_description_path,
                                            indices=indices,
                                            type_model_desc='gru')

    tmp = pad_sequence(tensor_lists, batch_first=True)
    desc_ = pack_padded_sequence(tmp,
                                 torch.tensor([len(x) for x in tensor_lists]),
                                 batch_first=True,
                                 enforce_sorted=False)
    return desc_


def get_list_tensor_description_self_attention(data_description_path, indices):

    tensor_lists = get_related_descriptions(data_description_path=data_description_path,
                                            indices=indices,
                                            type_model_desc="self_attention")

    desc_ = pad_sequence(tensor_lists, batch_first=True)

    lengths = torch.tensor([len(x) for x in tensor_lists])
    max_length = lengths.max().item()
    range_tensor = torch.arange(max_length)
    # Create a mask tensor by comparing the range tensor to the length of each unpadded tensor
    mask = range_tensor.unsqueeze(0) < lengths.unsqueeze(1)
    mask = ~mask

    return desc_, mask


# class MILNCELoss(torch.nn.Module):
#     def __init__(self):
#         super(MILNCELoss, self).__init__()
#
#     def forward(self, pairwise_similarity):
#         nominator = pairwise_similarity * torch.eye(pairwise_similarity.shape[0]).cuda()
#         nominator = torch.logsumexp(nominator, dim=1)
#         denominator = torch.cat((pairwise_similarity,
#                                  pairwise_similarity.permute(1, 0)), dim=1).view(pairwise_similarity.shape[0], -1)
#         denominator = torch.logsumexp(denominator, dim=1)
#         return torch.mean(denominator - nominator)


def evaluate_model(model_descriptor, model_scene, data_description, data_scene, data_description_living, data_description_bedroom,
                   data_scene_living, data_scene_bedroom, type_room, no_vids, type_model_desc, section, indices):
    model_descriptor = model_descriptor.eval()
    model_scene = model_scene.eval()
    if type_room in ['bedroom', 'living']:
        if type_model_desc in ['gru', 'bigru']:
            data_description = get_list_tensor_description(data_description_path=data_description,
                                                           indices=indices)
            if section != 'entire':
                data_scene = data_scene[indices]
        elif type_model_desc == "self_attention":
            data_description, mask = get_list_tensor_description_self_attention(data_description_path=data_description,
                                                                                indices=indices)
            if section != 'entire':
                data_scene = data_scene[indices]
        elif type_model_desc == "mean":
            data_description = get_related_descriptions(data_description_path=data_description,
                                                        indices=indices,
                                                        type_model_desc='mean')
            if section != 'entire':
                data_scene = data_scene[indices]
        if type_model_desc == "self_attention":
            # mask = mask.repeat_interleave(n_heads, 0)
            output_description = model_descriptor(data_description.cuda(), mask.cuda())
        else:
            output_description = model_descriptor(data_description.cuda())

    output_scene = model_scene(data_scene.cuda())

    avg_rank_scene = 0
    ranks_scene = []
    avg_rank_description = 0
    ranks_description = []

    ndcg_10_list = []
    ndcg_entire_list = []

    for j, i in enumerate(output_scene):
        rank, sorted_list = create_rank(i, output_description, j)
        avg_rank_scene += rank
        ranks_scene.append(rank)

        if section == "test":
            desired_output_indexes = get_desired_relevant(j, no_vids, "test", type_room)
            ndcg_10 = dcg_idcg_calculator(10, desired_output_indexes, sorted_list)
            ndcg = dcg_idcg_calculator("entire", desired_output_indexes, sorted_list)
            if ndcg_10 is not None:
                ndcg_10_list.append(ndcg_10)
            if ndcg is not None:
                ndcg_entire_list.append(ndcg)

    for j, i in enumerate(output_description):
        rank, sorted_list = create_rank(i, output_scene, j)
        avg_rank_description += rank
        ranks_description.append(rank)

        if section == "test":
            desired_output_indexes = get_desired_relevant(j, no_vids, "test", type_room)
            ndcg_10 = dcg_idcg_calculator(10, desired_output_indexes, sorted_list)
            ndcg = dcg_idcg_calculator("entire", desired_output_indexes, sorted_list)
            if ndcg_10 is not None:
                ndcg_10_list.append(ndcg_10)
            if ndcg is not None:
                ndcg_entire_list.append(ndcg)

    ranks_scene = np.array(ranks_scene)
    ranks_description = np.array(ranks_description)

    n_q = len(output_scene)
    ds_r1 = 100 * len(np.where(ranks_scene < 1)[0]) / n_q
    ds_r5 = 100 * len(np.where(ranks_scene < 5)[0]) / n_q
    ds_r10 = 100 * len(np.where(ranks_scene < 10)[0]) / n_q
    ds_medr = np.median(ranks_scene) + 1
    ds_meanr = ranks_scene.mean() + 1

    n_q = len(output_description)
    sd_r1 = 100 * len(np.where(ranks_description < 1)[0]) / n_q
    sd_r5 = 100 * len(np.where(ranks_description < 5)[0]) / n_q
    sd_r10 = 100 * len(np.where(ranks_description < 10)[0]) / n_q
    sd_medr = np.median(ranks_description) + 1
    sd_meanr = ranks_description.mean() + 1

    ds_out, sc_out = "", ""
    for mn, mv in [["R@1", ds_r1],
                   ["R@5", ds_r5],
                   ["R@10", ds_r10],
                   ["median rank", ds_medr],
                   ["mean rank", ds_meanr],
                   ]:
        ds_out += f"{mn}: {mv:.4f}   "

    for mn, mv in [("R@1", sd_r1),
                   ("R@5", sd_r5),
                   ("R@10", sd_r10),
                   ("median rank", sd_medr),
                   ("mean rank", sd_meanr),
                   ]:
        sc_out += f"{mn}: {mv:.4f}   "

    print(section + " data: ")
    print("Scenes ranking: " + ds_out)
    print("Descriptions ranking: " + sc_out)
    if section == "test":
        avg_ndcg_10_entire = 100 * sum(ndcg_10_list) / len(ndcg_10_list)
        avg_ndcg_entire = 100 * sum(ndcg_entire_list) / len(ndcg_entire_list)
    else:
        avg_ndcg_10_entire = -1
        avg_ndcg_entire = -1

    model_descriptor = model_descriptor.train()
    model_scene = model_scene.train()

    return ds_r1, ds_r5, ds_r10, sd_r1, sd_r5, sd_r10, avg_ndcg_10_entire, avg_ndcg_entire, ds_medr, sd_medr


def cosine_sim(im, s):
    '''cosine similarity between all the image and sentence pairs
    '''
    inner_prod = im.mm(s.t())
    im_norm = torch.sqrt((im ** 2).sum(1).view(-1, 1) + 1e-18)
    s_norm = torch.sqrt((s ** 2).sum(1).view(1, -1) + 1e-18)
    sim = inner_prod / (im_norm * s_norm)
    return sim


def print_results(avg_ndcg_10_entire, avg_ndcg_entire, avg_r_10_entire, avg_r_1_entire, avg_r_50_entire,avg_r_5_entire):
    print("avg_r_1_entire:", avg_r_1_entire)
    print("avg_r_5_entire:", avg_r_5_entire)
    print("avg_r_10_entire:", avg_r_10_entire)
    print("avg_r_50_entire:", avg_r_50_entire)
    print("avg_ndcg_10_entire:", avg_ndcg_10_entire)
    print("avg_ndcg_entire:", avg_ndcg_entire)


def analyze_ndcg_queries(scene_result, entire_descriptor, desired_output_indexes):
    similarity = torch.nn.functional.cosine_similarity(entire_descriptor, scene_result, dim=1)
    similarity = similarity.squeeze()
    sorted_indices = torch.argsort(similarity, descending=True)
    sorted_indices = sorted_indices.tolist()

    ndcg_10 = dcg_idcg_calculator(10, desired_output_indexes, sorted_indices)
    ndcg = dcg_idcg_calculator("entire", desired_output_indexes, sorted_indices)

    return ndcg, ndcg_10


def get_embeddings(sentence, device, tokenizer, model_bert):
    with torch.no_grad():
        inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        inputs = inputs.to(device)
        outputs = model_bert(**inputs)
        sentence_embeddings = outputs.last_hidden_state
        sentence_embeddings = torch.mean(sentence_embeddings, dim=1)
        obtained_tensor = sentence_embeddings
    return obtained_tensor


def evaluate_style_queries(model_descriptor, model_scene, data_scene, data_scene_living, data_scene_bedroom, indices, type_room, desc_model_type, batch_size):
    if type_room not in ['combined']:
        with open(
                '../descriptions/sentence_features/desc_strings_' + type_room + '.pkl',
                'rb') as f:
            scenes_info = pickle.load(f)
            scenes_info = [scenes_info[i] for i in indices.numpy()]

    entire_theme_list = []
    entire_style_list = []
    entire_material_list = []

    containing_index = {'material': {}, 'style': {}, 'theme': {}}

    number_of_queries = 0

    for i, s in enumerate(scenes_info):
        theme_list = []
        style_list = []
        material_list = []

        no_object = sum(i['number'] for i in s)

        for o in s:
            if 'theme' in o.keys():
                if o['theme'] is not None and o['theme'] != 'Others':
                    theme_list.extend([o['theme']] * o['number'])
            if 'style' in o.keys():
                if o['style'] is not None and o['style'] != 'Others':
                    style_list.extend([o['style']] * o['number'])
            if 'material' in o.keys():
                if o['material'] is not None and o['material'] != 'Others':
                    material_list.extend([o['material']] * o['number'])

        themes_set = set(theme_list)
        style_set = set(style_list)
        material_set = set(material_list)

        for t in themes_set:
            c = theme_list.count(t)
            if c * 2 >= no_object:
                entire_theme_list.append(t)
                if t in containing_index['theme']:
                    containing_index['theme'][t].append(i)
                else:
                    containing_index['theme'][t] = [i]

        for t in style_set:
            c = style_list.count(t)
            if c * 2 >= no_object:
                entire_style_list.append(t)
                if t in containing_index['style']:
                    containing_index['style'][t].append(i)
                else:
                    containing_index['style'][t] = [i]

        for t in material_set:
            c = material_list.count(t)
            if c * 2 >= no_object:
                entire_material_list.append(t)
                if t in containing_index['material']:
                    containing_index['material'][t].append(i)
                else:
                    containing_index['material'][t] = [i]

    print(set(entire_theme_list))
    print(set(entire_material_list))
    print(set(entire_style_list))

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    if type_room not in ['combined']:
        data_scene = data_scene[indices]
    else:
        data_scene = get_scene_data(data_scene_living=data_scene_living,
                                    data_scene_bedroom=data_scene_bedroom,
                                    indices=indices)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    model_bert.to(device)

    r_50_entire = []
    r_10_entire = []
    r_5_entire = []
    r_1_entire = []

    ndcg_10_entire = []
    ndcg_entire = []

    model_descriptor = model_descriptor.eval()
    model_scene = model_scene.eval()

    for t in entire_theme_list:
        sent = "I look for a room with " + t + " theme."
        embedding = get_embeddings(sent, device, tokenizer, model_bert)
        number_of_queries += 1
        if desc_model_type == 'mean':
            embedding = embedding.repeat(batch_size, 1)
        output_description = model_descriptor(embedding)
        output_scene = model_scene(data_scene.cuda())
        if desc_model_type == 'mean':
            output_description = output_description[0]
        r_50, r_10, r_5, r_1 = create_percent_queries(output_description, output_scene, containing_index['theme'][t])

        r_50_entire.append(r_50)
        r_10_entire.append(r_10)
        r_5_entire.append(r_5)
        r_1_entire.append(r_1)

        ndcg, ndcg_10 = analyze_ndcg_queries(output_description, output_scene, containing_index['theme'][t])
        if ndcg_10 is not None:
            ndcg_10_entire.append(ndcg_10)
        if ndcg is not None:
            ndcg_entire.append(ndcg)

    for t in entire_style_list:
        sent = "I look for a room with " + t + " style."
        embedding = get_embeddings(sent, device, tokenizer, model_bert)
        number_of_queries += 1
        if desc_model_type == 'mean':
            embedding = embedding.repeat(batch_size, 1)
        output_description = model_descriptor(embedding)
        output_scene = model_scene(data_scene.cuda())
        if desc_model_type == 'mean':
            output_description = output_description[0]
        r_50, r_10, r_5, r_1 = create_percent_queries(output_description, output_scene, containing_index['style'][t])

        r_50_entire.append(r_50)
        r_10_entire.append(r_10)
        r_5_entire.append(r_5)
        r_1_entire.append(r_1)

        ndcg, ndcg_10 = analyze_ndcg_queries(output_description, output_scene, containing_index['style'][t])
        if ndcg_10 is not None:
            ndcg_10_entire.append(ndcg_10)
        if ndcg is not None:
            ndcg_entire.append(ndcg)

    for t in entire_material_list:
        sent = "I look for a room with " + t + " material."
        embedding = get_embeddings(sent, device, tokenizer, model_bert)
        number_of_queries += 1
        if desc_model_type == 'mean':
            embedding = embedding.repeat(batch_size, 1)
        output_description = model_descriptor(embedding)
        output_scene = model_scene(data_scene.cuda())
        if desc_model_type == 'mean':
            output_description = output_description[0]

        r_50, r_10, r_5, r_1 = create_percent_queries(output_description, output_scene, containing_index['material'][t])

        r_50_entire.append(r_50)
        r_10_entire.append(r_10)
        r_5_entire.append(r_5)
        r_1_entire.append(r_1)

        ndcg, ndcg_10 = analyze_ndcg_queries(output_description, output_scene, containing_index['material'][t])
        if ndcg_10 is not None:
            ndcg_10_entire.append(ndcg_10)
        if ndcg is not None:
            ndcg_entire.append(ndcg)

    avg_r_50_entire = 100 * sum(r_50_entire) / len(r_50_entire)
    avg_r_10_entire = 100 * sum(r_10_entire) / len(r_10_entire)
    avg_r_5_entire = 100 * sum(r_5_entire) / len(r_5_entire)
    avg_r_1_entire = 100 * sum(r_1_entire) / len(r_1_entire)
    avg_ndcg_10_entire = 100 * sum(ndcg_10_entire) / len(ndcg_10_entire)
    avg_ndcg_entire = 100 * sum(ndcg_entire) / len(ndcg_entire)
    print("Number of Style Queries", number_of_queries)

    print_results(avg_ndcg_10_entire, avg_ndcg_entire, avg_r_10_entire, avg_r_1_entire, avg_r_50_entire, avg_r_5_entire)
    model_descriptor = model_descriptor.train()
    model_scene = model_scene.train()
    return avg_r_1_entire, avg_r_5_entire, avg_r_10_entire, avg_ndcg_10_entire, avg_ndcg_entire


def process_captions(captions):
    new_captions = []
    verbs = []
    nlp = spacy.load("en_core_web_sm")
    for c in captions:
        tmp_sent = "I " + c
        split_sent = c.split()
        res = nlp(tmp_sent)
        # print("* " * 30)
        for token in res:
            # print(token.text, " ", token.pos_)
            if token.pos_ == "VERB":
                verb = token.text
                if verb not in verbs:
                    verbs.append(verb)
                if verb[-1] == 's':
                    continue
                if split_sent.index(verb) > 1:
                    if split_sent[split_sent.index(verb) - 1] != "and":
                        continue
                if verb[-1] == 't' and verb != "heat":
                    ing_form = Word(verb).lemmatize('v') + 't' + 'ing'
                elif verb[-1] == 'e':
                    ing_form = Word(verb[:-1]).lemmatize('v') + 'ing'
                else:
                    ing_form = Word(verb).lemmatize('v') + 'ing'

                split_sent[split_sent.index(verb)] = ing_form
        new_captions.append(' '.join(split_sent))

    return new_captions, verbs


def evaluate_video_queries(model_descriptor, model_scene, data_scene, data_scene_living, data_scene_bedroom, indices, no_of_videos, type_room, desc_model_type, batch_size):
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    model_bert.to(device)

    if type_room not in ['combined']:
        data_scene = data_scene[indices]
    else:
        data_scene = get_scene_data(data_scene_living=data_scene_living,
                                    data_scene_bedroom=data_scene_bedroom,
                                    indices=indices)

    # processing video related queries
    video_captions = open("../data_youcook2/captions.pkl", 'rb')
    video_captions = pickle.load(video_captions)
    video_captions = video_captions[:no_of_videos]
    processed_captions, _ = process_captions(video_captions)

    r_50_entire = []
    r_10_entire = []
    r_5_entire = []
    r_1_entire = []

    ndcg_10_entire = []
    ndcg_entire = []

    model_descriptor = model_descriptor.eval()
    model_scene = model_scene.eval()

    if type_room not in ['combined']:
        video_indexes = [x % no_of_videos for x in indices.tolist()]
    else:
        video_indexes = []
        with open('indices/indices_combined_guidance.pkl', 'rb') as f:
            indices_guidance = pickle.load(f)
        for idx in indices:
            video_indexes.append(indices_guidance[idx]['index'] % no_of_videos)
    set_videos = set(video_indexes)
    print("Video Queries len_set: ", len(set_videos))

    for t in set_videos:
        sent = "I look for a room in which the TV showing  " + processed_captions[t] + "."
        embedding = get_embeddings(sent, device, tokenizer, model_bert)
        if desc_model_type == 'mean':
            embedding = embedding.repeat(batch_size, 1)
        output_description = model_descriptor(embedding)
        output_scene = model_scene(data_scene.cuda())
        desired = [i for i, x in enumerate(video_indexes) if x == t]
        if desc_model_type == 'mean':
            output_description = output_description[0]
        r_50, r_10, r_5, r_1 = create_percent_queries(output_description, output_scene, desired)

        r_50_entire.append(r_50)
        r_10_entire.append(r_10)
        r_5_entire.append(r_5)
        r_1_entire.append(r_1)

        ndcg, ndcg_10 = analyze_ndcg_queries(output_description, output_scene, desired)

        if ndcg_10 is not None:
            ndcg_10_entire.append(ndcg_10)
        if ndcg is not None:
            ndcg_entire.append(ndcg)

    avg_r_50_entire = 100 * sum(r_50_entire) / len(r_50_entire)
    avg_r_10_entire = 100 * sum(r_10_entire) / len(r_10_entire)
    avg_r_5_entire = 100 * sum(r_5_entire) / len(r_5_entire)
    avg_r_1_entire = 100 * sum(r_1_entire) / len(r_1_entire)

    avg_ndcg_10_entire = -1
    avg_ndcg_entire = -1
    if len(ndcg_10_entire) != 0:
        avg_ndcg_10_entire = sum(ndcg_10_entire) / len(ndcg_10_entire)
    if len(ndcg_entire) != 0:
        avg_ndcg_entire = sum(ndcg_entire) / len(ndcg_entire)

    print_results("avg_ndcg_10_entire", "avg_ndcg_entire", avg_r_10_entire, avg_r_1_entire, avg_r_50_entire, avg_r_5_entire)

    model_descriptor = model_descriptor.train()
    model_scene = model_scene.train()
    return avg_r_1_entire, avg_r_5_entire, avg_r_10_entire, avg_ndcg_10_entire, avg_ndcg_entire


def evaluate_distance_queries(model_descriptor, model_scene, data_scene, data_scene_living, data_scene_bedroom, indices, type_room, desc_model_type, batch_size):
    if type_room not in ['combined']:
        with open(
                '../descriptions/sentence_features/entire_positional_info_' + type_room + '.pkl',
                'rb') as f:
            dist_info = pickle.load(f)
        dist_info = [dist_info[i] for i in indices.numpy()]
    else:
        with open(
                '../descriptions/sentence_features/entire_positional_info_' + 'living' + '.pkl',
                'rb') as f:
            dist_info_living = pickle.load(f)
        with open(
                '../descriptions/sentence_features/entire_positional_info_' + 'bedroom' + '.pkl',
                'rb') as f:
            dist_info_bedroom = pickle.load(f)

            dist_info = []
            with open('indices/indices_combined_guidance.pkl', 'rb') as f:
                indices_guidance = pickle.load(f)
            for idx in indices:
                retrieve_orig = indices_guidance[idx]
                if retrieve_orig['type'] == 'living':
                    dist_info.append(dist_info_living[retrieve_orig['index']])
                elif retrieve_orig['type'] == 'bedroom':
                    dist_info.append(dist_info_bedroom[retrieve_orig['index']])

    unique_list = []
    for i in dist_info:
        for j in i["info"]:
            is_found = False
            for o in unique_list:
                if o['obj1'] == j['obj1'] and o['obj2'] == j['obj2'] and o['rel'] == j['rel']:
                    o["indexes"].append(i['scene_index'])
                    is_found = True
                    break
                elif o['obj1'] == j['obj2'] and o['obj2'] == j['obj1'] and o['rel'] == j['rel']:
                    o["indexes"].append(i['scene_index'])
                    is_found = True
                    break
            if not is_found:
                new_dict = {"obj1": j['obj1'], "rel": j['rel'], "obj2": j['obj2'], "indexes": [i['scene_index']]}
                unique_list.append(new_dict)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    if type_room not in ['combined']:
        data_scene = data_scene[indices]
    else:
        data_scene = get_scene_data(data_scene_living=data_scene_living,
                                    data_scene_bedroom=data_scene_bedroom,
                                    indices=indices)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    model_bert.to(device)

    r_50_entire = []
    r_10_entire = []
    r_5_entire = []
    r_1_entire = []

    ndcg_10_entire = []
    ndcg_entire = []

    temp_hist_dist = []

    model_descriptor = model_descriptor.eval()
    model_scene = model_scene.eval()

    for t in unique_list:
        if len(t["indexes"]) >= 10:
            temp_hist_dist.append(len(t["indexes"]))
            sent = "I look for a room in which " + t['obj1'] + " is " + t['rel'] + " " + t['obj2'] + "."
            embedding = get_embeddings(sent, device, tokenizer, model_bert)
            if desc_model_type == 'mean':
                embedding = embedding.repeat(batch_size, 1)
            output_description = model_descriptor(embedding)
            output_scene = model_scene(data_scene.cuda())
            if desc_model_type == 'mean':
                output_description = output_description[0]
            r_50, r_10, r_5, r_1 = create_percent_queries(output_description, output_scene, t["indexes"])

            r_50_entire.append(r_50)
            r_10_entire.append(r_10)
            r_5_entire.append(r_5)
            r_1_entire.append(r_1)

            ndcg, ndcg_10 = analyze_ndcg_queries(output_description, output_scene, t["indexes"])

            if ndcg_10 is not None:
                ndcg_10_entire.append(ndcg_10)
            if ndcg is not None:
                ndcg_entire.append(ndcg)

    print("Distance Queries ")

    print("len length>10: ", len(temp_hist_dist))
    print("max length>10: ", max(temp_hist_dist))
    print("min length>10: ", min(temp_hist_dist))

    avg_r_50_entire = 100 * sum(r_50_entire) / len(r_50_entire)
    avg_r_10_entire = 100 * sum(r_10_entire) / len(r_10_entire)
    avg_r_5_entire = 100 * sum(r_5_entire) / len(r_5_entire)
    avg_r_1_entire = 100 * sum(r_1_entire) / len(r_1_entire)
    avg_ndcg_10_entire = 100 * sum(ndcg_10_entire) / len(ndcg_10_entire)
    avg_ndcg_entire = 100 * sum(ndcg_entire) / len(ndcg_entire)

    print_results(avg_ndcg_10_entire, avg_ndcg_entire, avg_r_10_entire, avg_r_1_entire, avg_r_50_entire, avg_r_5_entire)

    model_descriptor = model_descriptor.train()
    model_scene = model_scene.train()

    return avg_r_1_entire, avg_r_5_entire, avg_r_10_entire, avg_ndcg_10_entire, avg_ndcg_entire


def plot_procedure(train_losses, val_losses):
    plt.plot(train_losses, color='red', label='Training loss')
    plt.plot(val_losses, color='blue', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.show()
