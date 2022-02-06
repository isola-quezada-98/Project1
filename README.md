"""DOMAIN ADAPTATION IN SEMANTIC SEGMENTATION""" 
How to use these files


Place all dataset files (https://mega.nz/file/ERkiQBaY#h-wktK7U7MpIG5nf-rMWF7d76NEM5ae_MrAmELftNR0) as a folder named "data" 

Checkpoints for all models are already available, and the correspoinding semantic segmentation outputs can be seen in "best_pseudolabels{model_type}", so there is no need to run any code to obtain them

The colorized ground truth labels for Cityscapes are also availabe in "color_labels"

run "train_base.py" to train BiSeNet with Cityscapes
run "train_DomainAdapt.py" to train the domain adaptation network, use lines 406-407 to select between normal and lightweight models
run "train_FDA.py" to train FDA (needs pytorch downgrade due to compatibility issues: torch==1.7.1 torchvision==0.8.2 )
run "generate_labels.py" to create segmentation output labels on the validation set (use line 9 to select the model, will use the checkpoint with the best loss)
run "flops.py" to calculate model flops
run "label_colorized.py" to colorize the cityscapes ground truth labels (ALREADY DONE, see above)

Further comments:
Dice loss is not available as it was not used, so the checkpoints called "best_dice_loss" and "latest_dice_loss" actually come from training done using cross-entropy loss
