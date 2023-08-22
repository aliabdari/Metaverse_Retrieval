import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--no_sample", help="number of samples based on the type of the room",
                    default=3384, required=False)
args = parser.parse_args()
video_names = open("./video_names.pkl", 'rb')
video_names = pickle.load(video_names)

no_selected_vids = 25

for i in range(args.no_sample):
    with open('./videos/vid_' + str(i) + '.txt', 'w') as f:
        f.write(video_names[i % no_selected_vids])
    print(i % no_selected_vids)
