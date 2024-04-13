#import stuff
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def main():
    import numpy as np

    # Load sampled images from the NumPy array file
    samples_file = 'samples_<shape>.npz'  # Replace '<shape>' with the actual shape
    data = np.load(samples_file)

    # Access sampled images
    sampled_images = data['arr']  # For non-class conditioned samples

    # If class conditioning is enabled
    if 'label_arr' in data:
        sampled_labels = data['label_arr']

    
    #import the labels from test_labels
    #import the model results from model_results

    #wite code for getting the truth labels




# Now you can use the sampled_images and sampled_labels arrays as needed



#get the data

#get the model result
#here you have to find the rigth model, that is trained, at the moment when running
#classifier_sample_known.py, the model is loaded from the path, so you have to find the right model, as it looks
#for model.pt, but there are different names
#det ligner at de første fem har som har samme navn, er samme model. De har også samme størrelse
#

#get the ground truth



def dice_score(output, ground_truth):

    intersection = np.logical_and(output, ground_truth)
    return 2. * intersection.sum() / (output.sum() + ground_truth.sum())

def iou_score(output, ground_truth):
    
        intersection = np.logical_and(output, ground_truth)
        union = np.logical_or(output, ground_truth)
        return intersection.sum() / union.sum()

def auroc_score(output, ground_truth):
    #read about how this works
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(ground_truth, output)


def threshold(output, threshold=0.5):
    #make the output binary of the images, as in 1 and 0
    #should the input be numpy?
    return output > threshold #does this work? doesn't it give true/false?

if __name__ == '__main__':
    main()






