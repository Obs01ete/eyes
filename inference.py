import sys
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch

from train import Classifier, LATENT_SIZE


def main():
    model = Classifier(LATENT_SIZE)
    state_dict = torch.load("classifier.pth")
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()

    folder = sys.argv[1]
    print(folder)
    file_list = glob.glob(folder + "/*.*")
    predictions = []
    for file in tqdm(file_list):
        image = Image.open(file)
        image_np = np.asarray(image)
        tensor_image = (1 / 255 * torch.tensor(image_np).float())
        tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
        tensor_image = tensor_image.cuda()
        pred = model(tensor_image)
        hard_pred = (pred > 0.5).long()
        pred_item = hard_pred.item()
        predictions.append(pred_item)

    with open("result.csv", "w") as f:
        for file, pred in zip(file_list, predictions):
            file = file.replace("\\", "/")
            f.write(f"{file},{pred}\n")



if __name__ == "__main__":
    main()