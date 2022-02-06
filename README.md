"""DOMAIN ADAPTATION IN SEMANTIC SEGMENTATION""" 
How to use these files



Checkpoints for all models are already available, and the correspoinding semantic segmentation outputs can be seen in "./data/Cityscapes/best_pseudolabels{model_type}", so there is no need to run any code to obtain them
The colorized ground truth labels for Cityscapes are also availabe in "./data/Cityscapes/color_labels"

run "train_base.py" to train BiSeNet with Cityscapes
run "train_DomainAdapt.py" to train the domain adaptation network, use lines 406-407 to select between normal and lightweight models
run "train_FDA.py" to train FDA (needs pytorch downgrade due to compatibility issues: torch==1.7.1 torchvision==0.8.2 )
run "generate_labels.py" to create segmentation output labels on the validation set (use line 9 to select the model, will use the checkpoint with the best loss)
run "flops.py" to calculate model flops
run "label_colorized.py" to colorize the cityscapes ground truth labels

Further comments:
Dice loss is not available as it was not used, so the checkpoints called "best_dice_loss" and "latest_dice_loss" actually come from training done using cross-entropy loss
