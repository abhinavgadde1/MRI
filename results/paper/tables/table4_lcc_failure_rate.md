| model | n_heldout | n_lcc_dice_zero | rate | cases | mechanism |
| --- | --- | --- | --- | --- | --- |
| v1 (brats_pretrain) | 60 | 2 | 2/60 | BraTS20_Training_146; BraTS20_Training_147 | Largest-CC heuristic keeps a larger false-positive blob; true-tumor component is discarded (raw Dice already low). |
| v2 (brats_v2_loss_fix) | 60 | 2 | 2/60 | BraTS20_Training_288; BraTS20_Training_271 | Same largest-CC failure mode (confirmed on 288/271: raw∩GT>0, LCC∩GT=0). |
| v3 (brats_scale_full) | 60 | 1 | 1/60 | BraTS20_Training_259 | Same largest-CC failure mode (confirmed on 259: raw∩GT>0, LCC∩GT=0). |
