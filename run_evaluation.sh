MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"

#!/bin/sh
#BSUB -J eval
#BSUB -o outputs/%J.out
#BSUB -e outputs/%J.err
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=20G]"
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s233039@dtu.dk
#BSUB -N
#BSUB -W 24:00
# end of BSUB options

nvidia-smi
# Load the cuda module
module load cuda/11.0

# activate the virtual environment 
source ./adlcv/bin/activate

python scripts/evaluate_model.py --data_dir ~/Desktop/ADLCV-AnomalyDetection/data/brats/val --model_path ./results/brats2update050000Lasse2.pt --classifier_path ./results/modelbratsclass14999Lasse.pt $CLASSIFIER_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

