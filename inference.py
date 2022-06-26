import torch
import os
import tqdm
from torchvision.transforms import ToPILImage
from PIL import Image

def save_image(pred, file_name):
    pred = ToPILImage()(pred)
    pred.save(file_name)


def inference(model, test_loader, output_dir):

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader, total=len(test_loader.dataset)):
            image, file_name = data["image"], data["file_name"]
            pred = model(image)
            pred = torch.argmax(pred, dim=1)

            save_image(pred, os.path.join(output_dir, file_name + "_prediction.png"))


if __name__ == "__main__":

    inference(model, test_loader, output_dir)