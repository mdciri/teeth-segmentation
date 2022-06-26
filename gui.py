import os

from tkinter import *
from tkinter import filedialog  
from PIL import ImageTk,Image  

import torch
import numpy as np

import monai.transforms as mt
from monai.networks.nets import UNet
from monai.networks.layers import Norm, Act
from monai.visualize.utils import blend_images

root = Tk()  
root.geometry("600x800")
root.title("Teeth segmentation")

device = "mps:0"
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    act = Act.LEAKYRELU
).to(device)
checkpoint = torch.load("./models/checkpoint1/model.pt")
model.load_state_dict(checkpoint["model_state_dict"])

def mask_img(img):

    img = np.asarray(img, dtype=np.float32)

    msk_path = "./Segmentation/maxillomandibular/" + root.filename.split("/")[-1].split(".")[0] + ".jpg"
    msk = np.asarray(Image.open(msk_path).convert("1"), dtype="float32")
    return np.expand_dims(img*msk, axis=0)

def save_img(img_pil):

    preds_dir = "./Predictions"
    if os.path.exists(preds_dir) == False:
        os.makedirs(preds_dir)

    out_name = os.path.join(preds_dir, root.filename.split("/")[-1].split(".")[0] + "_pred.png")
    img_pil.save(out_name)
    save_label = Label(root, text=f"image saved in {out_name}").pack()

def predict(img_pil):
    global overlay

    transforms = mt.compose.Compose(
        [
            mt.NormalizeIntensity(
                nonzero=True, 
                channel_wise=True
            ),
            mt.Resize(
                spatial_size=(256, 512), 
                mode="bilinear"
            ),
            mt.ToTensor(
                dtype=torch.float32
            )
        ]
    )

    img = mask_img(img_pil)
    img_preproc = transforms(img).to(device)
    pred = model(img_preproc.unsqueeze(0))
    pred = torch.nn.Softmax(dim=1)(pred)
    pred = torch.argmax(pred, dim=1).cpu().numpy()

    img_resized = img_pil.resize((512, 256))
    img_resized = np.asarray(img_resized)/255
    img_resized = np.expand_dims(img_resized, 0)

    # print(img_resized.shape, pred.shape)

    overlay = blend_images(img_resized, pred)
    overlay = np.transpose(overlay, (1, 2, 0))*255
    overlay_pil = Image.fromarray(overlay.astype(np.uint8))
    overlay = ImageTk.PhotoImage(overlay_pil)
    overlay_label = Label(image=overlay).pack()

    save_button = Button(root, text="save", command=lambda: save_img(overlay_pil)).pack()

def open():
    global image
    root.filename = filedialog.askopenfilename(
        initialdir="./Radiographs",
        title="Radiographs directory",
    )
    #my_label = Label(root, text=root.filename).pack()
    image_pil = Image.open(root.filename).convert("L")
    image = ImageTk.PhotoImage(image_pil.resize((512, 256)))
    image_label = Label(image=image).pack()

    predict_button = Button(root, text="predict", command=lambda: predict(image_pil)).pack()


open_button = Button(root, text="open file", command=open).pack()



root.mainloop() 