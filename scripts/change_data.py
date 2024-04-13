import os
import nibabel as nib
import numpy as np
"""
Changes the data structure from having one file for each patient containing 155 slices to 
having one file for each slice so the loading is faster

You have to specify starting directory of the data now


"""
def load_data(root, files):
    out = []
    files.sort()
    print(files)
    for  file in files:
        out.append(nib.load(os.path.join(root, file)).get_fdata())

    out = np.stack(out)
    return out


def create_files(data, path):
    for index in range(data.shape[3]):
        image = nib.Nifti1Image(data[:,:,:, index], np.eye(4), )
        number = F"{0 if index+1 < 100 else ''}{0 if index+1 < 10 else ''}{index+1}"
        nib.save(image, F"{path}/brats_train_{path[-3:]}_{number}.nii.gz")

def main(directory_from, directory_to):
    tries = 0
    for root, dirs, files in os.walk(directory_from):
        if not dirs:
            tries += 1
            data = load_data(root, files)
            dirname = root.split('/')[-1]
            print(dirname)
            dirname = os.path.join(directory_to, dirname)
            try:
                os.mkdir(dirname)
            except FileExistsError:
                pass
            create_files(data, dirname)





if __name__ == "__main__":
    directory_to_data = '/zhome/af/9/203285/Desktop/ADLCV-AnomalyDetection/data/brats'
    directory_from = os.path.join(directory_to_data, 'alltraining')
    directory_to = os.path.join(directory_to_data, 'training')
    main(directory_from = directory_from, directory_to = directory_to)