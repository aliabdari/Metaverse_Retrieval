'''
This script aims to obtain relevant scenes for each scene, which would be used for evaluation step (NDCG)
'''
import pickle


def relevance_degree_position(positional_info_1, positional_info_2):
    mutual = 0
    for p1 in positional_info_1['info']:
        for p2 in positional_info_2['info']:
            if p1['obj1'] == p2['obj1'] and p1['obj2'] == p2['obj2'] and p1['rel'] == p2['rel']:
                mutual += 1
            elif p1['obj2'] == p2['obj1'] and p2['obj2'] == p1['obj1'] and p1['rel'] == p2['rel']:
                mutual += 1
    # union = len(positional_info_1['info']) + len(positional_info_2['info']) - mutual
    union = len(positional_info_1['info'])
    return mutual, union


def relevance_degree_objects(object_info_1, object_info_2):
    mutual = 0
    for p1 in object_info_1:
        for p2 in object_info_2:
            if p1['category'] == p2['category'] and p1['style'] == p2['style'] \
                    and p1['theme'] == p2['theme'] and p1['material'] == p2['material']:
                # mutual += max(p1['number'], p2['number'])
                mutual += p1['number']

    union = 0
    for p in object_info_1:
        union += p['number']
    # for p in object_info_2:
    #     union += p['number']

    return mutual, union


def get_desired_relevant(no_of_videos, section_, type_room):
    indices_pickle = open(f'indices_{type_room}.pkl', "rb")
    indices_pickle = pickle.load(indices_pickle)
    if section_ == "train":
        indices = indices_pickle["train"].tolist()
    elif section_ == "val":
        indices = indices_pickle["val"].tolist()
    elif section_ == "test":
        indices = indices_pickle["test"].tolist()
    else:
        # TODO the static numbers should be modified
        indices = range(3384)

    if type_room in ['living', 'bedroom']:
        with open('../descriptions/sentence_features/desc_strings_'
                  + type_room + '.pkl', 'rb') as f:
            objects_info = pickle.load(f)
        objects_info = [objects_info[i] for i in indices]

        with open('../descriptions/sentence_features/entire_positional_info_'
                  + type_room + '.pkl', 'rb') as f:
            dist_info = pickle.load(f)
        dist_info = [dist_info[i] for i in indices]

    desired_list_entire = []
    for idx1, s1 in enumerate(indices):
        print(s1)
        desired_list = [idx1]
        for idx2, s2 in enumerate(indices):
            if s1 == s2:
                continue
            mutual_position_no, union_position = relevance_degree_position(dist_info[idx1], dist_info[idx2])
            mutual_objects_no, union_objects = relevance_degree_objects(objects_info[idx1], objects_info[idx2])
            mutuals = mutual_objects_no + mutual_position_no
            unions = union_objects + union_position

            if s1 % no_of_videos == s2 % no_of_videos:
                mutuals += 1
            if mutuals / (unions + 1) >= .5:
                desired_list.append(s2)
        desired_list_entire.append(desired_list)
    return desired_list_entire


if __name__ == "__main__":
    no_vid = 25
    section = 'test'
    type_room = 'bedroom'
    desired_list_entire = get_desired_relevant(no_vid, section, type_room)
    with open('./relevances/relevance_' + str(no_vid) + '_' + section + '_' + type_room + '.pkl', 'wb') as f:
        pickle.dump(desired_list_entire, f)
    print("Finished")
