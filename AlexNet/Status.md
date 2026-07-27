# Current Implementation

Implementation like in the original paper https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

With kaiman initialization
with data augmentation. (see custum_datasets/imagenet_subset_dataset.py)

# Training

The training plateaux to 6.9 loss after 20 epochs, which is random guessing (log(1000) = 6.9)

The main issue is that the gradient vanishes to 0 and I have not found a way to keep it meaningful.
The gradient goes lower as we go deeper in the neural network. 

Tried:

- Batch Normalization instead of LocalResponseNorm
- Adam optimizer instead of sdg
- gradient_clip_val to 1.0 (for exploding gradient, no effect)
- accumulate_grad_batches 2 and 4
- kaiming_normal_ initialization instead of normal_ with 0.01 std dev 
