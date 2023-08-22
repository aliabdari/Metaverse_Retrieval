'''
This script would generate appropriate descriptions for each scenario.
Then related features for each description would be obtained using BERT model.
The previously generated descriptions exist in the /descriptions/descriptions_text
'''
import itertools
import os

import numpy as np
import random
from transformers import BertTokenizer, BertModel
import torch
import inflect
import matplotlib.pyplot as plt
import pickle
import spacy
from textblob import Word
from tqdm import tqdm
import argparse


def num_to_words(num):
    p = inflect.engine()
    return p.number_to_words(num)


def get_theme_desc(i):
    if i['theme'] is not None:
        return ", with " + i['theme'] + " theme"
    return ""


def get_theme(i):
    return i['theme']


def get_material_desc(i):
    material = get_material(i)
    if material is not None and material != "Others":
        return ", and " + material + " material"
    return ""


def get_material(i):
    return i['material']


def get_style_desc(i):
    style = get_style(i)
    if style != "Others":
        if 'style' in style:
            return " with " + style + " "
        else:
            return " with " + style + " style "
    return ""


def get_style(i):
    if i['style'] == "Vintage/Retro":
        return 'Vintage'
    return i['style']


def get_category(i):
    if i['category'] == "Footstool / Sofastool / Bed End Stool / Stool":
        return "stool"
    if i['category'] == "Lounge Chair / Cafe Chair / Office Chair":
        return "chair"
    if i['category'] == "Sideboard / Side Cabinet / Console Table":
        return "Sideboard"
    if i['category'] == "Drawer Chest / Corner cabinet":
        return "Corner cabinet"
    if i['category'] == "Corner/Side Table":
        return "Corner Table"
    if i['category'] == "Bookcase / jewelry Armoire":
        return "Bookcase Armoire"
    return i['category']


def get_transitional_word():
    words_list = ["Also,", "Moreover,", "Additionally,", "Furthermore,"]
    random_element = random.choice(words_list)
    return random_element


def get_final_word():
    words_list = ["Finally,", "Eventually,", "Ultimately,"]
    random_element = random.choice(words_list)
    return random_element


def get_verb():
    words_list = ["contains", "comprises", "includes"]
    random_element = random.choice(words_list)
    return random_element


def get_description(i, j):
    category = get_category(i[j])
    style = get_style_desc(i[j])
    material = get_material_desc(i[j])
    theme = get_theme_desc(i[j])
    bs = category
    if style != "":
        if "style" not in style:
            bs += style + "style"
        else:
            bs += style
    if material != "":
        bs += material
    if theme != "":
        bs += theme
    return bs


def analyze_distance(distance):
    if distance is None:
        return None
    if distance < (mean - (2 * std)):
        return "so close to "
    elif (mean - (2 * std)) <= distance < (mean - (.25 * std)):
        return "close to "
    elif (mean + (.25 * std)) <= distance < (mean + (2 * std)):
        return "far from "
    elif (mean + (2 * std)) <= distance:
        return "so far from "
    return None


def process_captions(captions):
    new_captions = []
    verbs = []
    nlp = spacy.load("en_core_web_sm")
    for c in captions:
        tmp_sent = "I " + c
        split_sent = c.split()
        res = nlp(tmp_sent)
        # print("* " * 30)
        # print(tmp_sent)
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
                # tmp_sent = tmp_sent.replace(verb, ing_form)
                # ing_form = conjugate(verb, tense='part')
                # tmp_sent2 = tmp_sent2.replace(verb, ing_form)
        # print(' '.join(split_sent))
        new_captions.append(' '.join(split_sent))

    return new_captions, verbs


def get_embeddings(description):
    tokenized_sentence = description.split('.')
    print(tokenized_sentence[-1])
    obtained_tensor = torch.empty(len(tokenized_sentence) - 1, 768)
    cnt = 0
    with torch.no_grad():
        for idx in range(len(tokenized_sentence) - 1):
            inputs = tokenizer(tokenized_sentence[idx], padding=True, truncation=True, return_tensors='pt')
            inputs = inputs.to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
            sentence_embeddings = torch.mean(embeddings, dim=1)
            obtained_tensor[cnt, :] = sentence_embeddings.cpu()
            cnt += 1
    return obtained_tensor


def create_db(room):
    objects = []
    for idx in range(len(room)):
        category = get_category(room[idx])
        style = get_style(room[idx])
        theme = get_theme(room[idx])
        material = get_material(room[idx])
        temp_dict = {"category": category, "style": style, "theme": theme,
                     "material": material, "number": 1}
        objects.append(temp_dict)

    return objects


def find_uniques(db):
    unique_db = []
    for i in db:
        found = False
        for j in unique_db:
            if not found and j['category'] == i['category'] and j['style'] == i['style'] \
                    and j['theme'] == i['theme'] and j['material'] == i['material']:
                j['number'] += 1
                found = True
        if not found:
            unique_db.append(i)

    return unique_db


def is_addable_to_comparisons(obj1, obj2, pairs_list):
    category1 = get_category(obj1)
    style1 = get_style(obj1)
    theme1 = get_theme(obj1)
    material1 = get_material(obj1)

    category2 = get_category(obj2)
    style2 = get_style(obj2)
    theme2 = get_theme(obj2)
    material2 = get_material(obj2)

    new_dict = {"category1": category1, "style1": style1, "theme1": theme1, "material1": material1,
                "category2": category2, "style2": style2, "theme2": theme2, "material2": material2}

    if len(pairs_list) == 0:
        pairs_list.append(new_dict)
    else:
        for pl in pairs_list:
            if pl['category1'] == category1 and pl['style1'] == style1 and pl['theme1'] == theme1 and \
                    pl['material1'] == material1:
                if pl['category2'] == category2 and pl['style2'] == style2 and pl['theme2'] == theme2 and \
                        pl['material2'] == material2:
                    return False, pairs_list
            elif pl['category1'] == category2 and pl['style1'] == style2 and pl['theme1'] == theme2 and \
                    pl['material1'] == material2:
                if pl['category2'] == category1 and pl['style2'] == style1 and pl['theme2'] == theme1 and \
                        pl['material2'] == material1:
                    return False, pairs_list
            elif category1 == category2 and style1 == style2 and theme1 == theme2 and material1 == material2:
                return False, pairs_list
        pairs_list.append(new_dict)
    return True, pairs_list


parser = argparse.ArgumentParser()
parser.add_argument("--type_room", help="type room",
                    default='bedroom', required=False)
parser.add_argument("--no_videos", help="Number of videos which would be distributed among scenarios",
                    default=50, required=False)
args = parser.parse_args()

type_room = args.type_room

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if type_room == 'bedroom':
    path_npy = './dataset_3dfront/Bedroom_train_val.npy'
    path_pickle = './dataset_3dfront/list_description_layouts_Bedroom_final.pkl'
else:
    path_npy = './dataset_3dfront/Livingroom_train_val.npy'
    path_pickle = './dataset_3dfront/list_description_layouts_Livingroom_final.pkl'

dataset = np.load(path_npy)
current_pickle = open(path_pickle, "rb")
pickle_contents = pickle.load(current_pickle)

'''
after using the pretrained model for scene feature extraction, on some of sample the model had a problem
Those samples are presented in the following lists, which would be ignored in the following procedures
'''
if type_room == 'bedroom':
    problematic_list = [25, 431, 708, 765, 970, 1007, 1650, 2550, 2679, 2859, 3206, 3319, 3366]
else:
    problematic_list = [105, 190, 850, 1165, 1592, 1700, 2116, 2201, 2206, 2247, 2539, 2672, 2817, 2893, 3014, 3249, 3388, 3446, 4532, 4644]

final_tensor = torch.empty(dataset.shape[0] - len(problematic_list), 1, 768)
final_tensor.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

model.to(device)

counter = -1

# load caption
video_captions = open("./data_youcook2/captions.pkl", 'rb')
video_captions = pickle.load(video_captions)
no_videos = args.no_videos
index_video = 0
processed_captions, _ = process_captions(video_captions[:no_videos])

# check the existing distances
list_of_distances = []
for j, i in enumerate(pickle_contents):
    if j in problematic_list:
        continue
    current_data = dataset[j, :, :]
    valid_abs_index = np.where(current_data[:, -1])[0]
    combos = itertools.combinations(valid_abs_index, 2)

    for combo in combos:
        if np.any(np.isnan(current_data[combo[0]][3:6])) or np.any(np.isnan(current_data[combo[1]][3:6])):
            print("Nan Found")
        else:
            distance = np.linalg.norm(current_data[combo[0]][3:6] - current_data[combo[1]][3:6])
            list_of_distances.append(distance)

list_of_distances.sort()
# plt.hist(list_of_distances, 100)
list_of_distances = np.array(list_of_distances)
std = np.std(list_of_distances, axis=0)
mean = np.mean(list_of_distances, axis=0)
print(len(list_of_distances))
# plt.show()

sentences_number = []
tokens_numbers = []
len_hist = []
processed_dbs_objects_list = []

entire_positional_info = []
entire_positional_info_complete = []

for j, i in tqdm(enumerate(pickle_contents), total=len(pickle_contents)):
    # print("idx = ", j)
    if j in problematic_list:
        continue
    positional_info = {"scene_index": j, "info": []}
    positional_info_complete = {"scene_index": j, "info": []}
    current_data = dataset[j, :, :]
    valid_abs_index = np.where(current_data[:, -1])[0]
    valid_abs_index = valid_abs_index.tolist()
    valid_abs_index.remove(79)
    valid_abs_index = np.array(valid_abs_index)
    combos = itertools.combinations(valid_abs_index, 2)

    db_objects = create_db(i)
    processed_db_objects = find_uniques(db_objects)
    processed_dbs_objects_list.append(processed_db_objects)

    number_of_sent = 0
    basic_sentence = "This room contains "
    for ii, jj in enumerate(processed_db_objects):
        if jj['number'] == 1:
            if number_of_sent == 0:
                basic_sentence += "one " + jj['category'] + get_style_desc(jj) \
                                  + get_theme_desc(jj) + get_material_desc(jj) + "."
            else:
                basic_sentence += " " + get_transitional_word() + " it " + get_verb() + " one " + jj['category'] \
                                  + get_style_desc(jj) + get_theme_desc(jj) + get_material_desc(jj) + "."
        else:
            word_of_num = num_to_words(jj['number'])
            if number_of_sent == 0:
                basic_sentence += word_of_num + " " + jj['category'] + get_style_desc(jj) \
                                  + get_theme_desc(jj) + get_material_desc(jj) + "."
            else:
                basic_sentence += " " + get_transitional_word() + " it " + get_verb() + " " + word_of_num + " " \
                                  + jj['category'] + get_style_desc(jj) + get_theme_desc(jj) + get_material_desc(
                    jj) + "."
        number_of_sent += 1

    # TV_Sentence
    basic_sentence += " " + get_final_word() + " this room contains a TV, showing " + processed_captions[
        index_video % no_videos] + "."
    # print(processed_captions[index_video % no_using_video_captions])
    index_video += 1

    positional_information = ""
    pairs_list = []
    first_index_sent = []

    # print("len i", len(i))
    # print("len valid abs", len(valid_abs_index))

    for combo in combos:
        obj_1_index = np.where(valid_abs_index == combo[0])[0][0]
        obj_2_index = np.where(valid_abs_index == combo[1])[0][0]

        obj_1_description = get_description(i, obj_1_index)
        obj_2_description = get_description(i, obj_2_index)

        if np.any(np.isnan(current_data[combo[0]][3:6])) or np.any(np.isnan(current_data[combo[1]][3:6])):
            print("Nan Found")
        else:
            is_enabled, pairs_list = is_addable_to_comparisons(i[obj_1_index], i[obj_2_index], pairs_list)
            if is_enabled:
                distance = np.linalg.norm(current_data[combo[0]][3:6] - current_data[combo[1]][3:6])
                distance_exp = analyze_distance(distance)
                if distance_exp is not None:
                    positional_info["info"].append({"obj1": get_category(i[obj_1_index]), "rel": distance_exp, "obj2": get_category(i[obj_2_index])})
                    positional_info_complete["info"].append({"obj1": obj_1_description, "rel": distance_exp, "obj2": obj_2_description})
                    if combo[0] in first_index_sent:
                        exp = " " + get_transitional_word() + " it is " + distance_exp + "the " + obj_2_description + "."
                    else:
                        first_index_sent.append(combo[0])
                        exp = " The " + obj_1_description + " is " + distance_exp + "the " + obj_2_description + "."
                    positional_information += exp

    entire_positional_info.append(positional_info)
    entire_positional_info_complete.append(positional_info_complete)

    tokens = (basic_sentence + positional_information).split()
    # tokens = (basic_sentence).split()
    num_tokens = len(tokens)
    # print("num_tokens = ", num_tokens)
    tokens_numbers.append(num_tokens)
    sentences_number.append(len((basic_sentence + positional_information).split('.'))-1)
    result = get_embeddings(basic_sentence + positional_information)
    if not os.path.exists('./descriptions/sentence_features/description_tensors_vid' + str(no_videos) + '_' + type_room):
        os.mkdir('./descriptions/sentence_features/description_tensors_vid' + str(no_videos) + '_' + type_room)
    counter += 1
    torch.save(result,
               './descriptions/sentence_features/description_tensors_vid' + str(no_videos) + '_' + type_room + '/desc_' + str(
                counter) + '.pt')
    len_hist.append(result.size()[0])

    if not os.path.exists(f'./descriptions/descriptions_text/description_strings_{type_room}_' + str(no_videos)):
        os.mkdir(f'./descriptions/descriptions_text/description_strings_{type_room}_' + str(no_videos))
    with open(f'./descriptions/descriptions_text/description_strings_{type_room}_' + str(no_videos) + '/desc_' + str(
              counter) + '.txt', 'w') as f:
        f.write(basic_sentence + positional_information)

    final_tensor[counter, :, :] = torch.mean(result, 0)

    # print(basic_sentence + positional_information)

print("MAX TOKENS", max(tokens_numbers))
print("MIN TOKENS", min(tokens_numbers))
print("MAX SENTENCES NUMBERS", max(sentences_number))
print("MIN SENTENCES NUMBERS", min(sentences_number))
print("AVG SENTENCES NUMBERS", sum(sentences_number)/len(sentences_number))
print("MAX LEN HIST", max(len_hist))
print("MIN LEN HIST", min(len_hist))
print("AVG LEN HIST", sum(len_hist)/len(len_hist))


tokens_numbers.sort()
plt.hist(tokens_numbers, 200)
plt.show()

sentences_number.sort()
plt.hist(sentences_number, 200)
plt.show()

saving_sub_dir = './descriptions/sentence_features'

with open(saving_sub_dir + f'/desc_strings_{type_room}.pkl', 'wb') as f:
    pickle.dump(processed_dbs_objects_list, f)
with open(saving_sub_dir + f'/entire_positional_info_{type_room}.pkl', 'wb') as f:
    pickle.dump(entire_positional_info, f)
with open(saving_sub_dir + '/entire_positional_info_complete.pkl', 'wb') as f:
    pickle.dump(entire_positional_info_complete, f)
# with open(saving_sub_dir + f'/processed_dbs_objects_list_{type_room}.pkl', 'wb') as f:
#     pickle.dump(processed_dbs_objects_list, f)

torch.device('cpu')
torch.save(final_tensor, './descriptions/sentence_features/final_mean_descriptions_vid_'
           + str(no_videos) + '_' + type_room + '.pt')
print("FINISHED")

