import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import pdb
from scipy import ndimage

"""
Order of seq types in files
Flair
Segmentation 
T1
T1CE
T2
"""

def load_file(filename):
    nib_img = nibabel.load(filename).get_fdata()
    index = np.array([0, 2, 3, 4])
    label = torch.tensor(nib_img[1, :, :])
    nib_img = torch.tensor(nib_img[index, :, :])
    image = torch.zeros(4, 256, 256)
    label_padded = torch.zeros(256, 256)
    image[:,8:-8,8:-8]=nib_img #pad to a size of (256,256), doesn' contain segmentation
    label_padded[8:-8,8:-8]=label
    #pdb.set_trace()
    if (image != 0).any(): # Check if the image is empty - if so normalization will give NaN.
        for index in range(image.size(0)):
            if image[index,:,:].max() != 0:
                image[index,:,:] = image[index,:,:]/image[index,:,:].max()
    #pdb.set_trace()
    return image, label_padded

class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=False):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  BraTS20_Training_001_XXX.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation

                  each of the file contains 155 slices -> thus making it 3D structure 
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.database = []
        for root, dirs, files in os.walk(self.directory):
            if not dirs:
                files.sort()
                for f in files:
                    self.database.append(os.path.join(root, f))

    def __getitem__(self, idx):
        filename = self.database[idx]
        number = filename.split('/')[-1].split('_')[2]
        image, label = load_file(filename)
        if label.max()>0:
            weak_label=1
        else:
            weak_label=0
        out_dict = {"y" : weak_label}
        return (image, out_dict, weak_label, label, number)

    def __len__(self):
        return len(self.database)


if __name__ == '__main__':
    data = BRATSDataset("/zhome/af/9/203285/Desktop/ADLCV-AnomalyDetection/data/brats/training")
    print(len(data))
    print(data[0][3].size())
    print(data[0][4])
