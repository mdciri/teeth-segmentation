
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class TuftsDataset(Dataset):
    """ Tufts Dataset
    """

    def __init__(
        self,
        data_list,
        masking = False,
        transform = None,
    ):
        super(TuftsDataset).__init__()

        self.data_list = data_list
        self.masking = masking
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, id):

        if self.masking == True:
            msk = np.asarray(Image.open(self.data_list[id]["msk"]).convert("1"), dtype="float32")
            seg = np.asarray(Image.open(self.data_list[id]["seg"]).convert("1"), dtype="float32")
            img = np.asarray(Image.open(self.data_list[id]["img"]).convert("L"), dtype="float32") * msk
        else:
            seg = np.asarray(Image.open(self.data_list[id]["seg"]).convert("1"), dtype="float32")
            img = np.asarray(Image.open(self.data_list[id]["img"]).convert("L"), dtype="float32")
        
        data = {
            "img": np.expand_dims(img, axis=0),
            "seg": np.expand_dims(seg, axis=0)
        }

        if self.transform is not None:
            data = self.transform(data)

        return data