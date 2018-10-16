------------------------------------------------------------------
Variational Auto-encoder Zero-short Learning (VZSL), Sept. 21st 2018
	---Tensorflow Implementation of the VZSL model proposed by Wang, Wenlin, et al.

Jiayu Wu, UCLA Statistics Dept., wujiayu86@outlook.com
	Templates from UCLA course STAT M232A. 
------------------------------------------------------------------

Variational Auto-encoder Zero-short Learning (VZSL) implementation in tensorflow on the AWA (Animal with Attributes) dataset.

Python scripts:

"main.py" --- Run training or load from previous 

"model_VZSL.py" --- Build the model and the training graph in tensor flow

"utils.py" --- Functions for data manipulation

"/data/AWA1/VGG19_feature/Process_features.py" --- Preprocess data by merging features in separate folders and files into a single .txt file.


Other files:

"/checkpoint" ---  Store checkpoints in the training process, manually renamed to record different optimization schemes. When loading or continuing previous training, rename to "*_feature".

"/data/AWA1" --- Dataset: VGG19 "features.txt" and "labels.txt" (raw data in "/feature"); class-labeling from 1 to 50 in "classes.txt"; continuous attributes table in "attributes.txt"; test/training split in "test_classes.txt" (proposed split in [3], used by default) and "test_classes_ss.txt" (standard split). (*Note that the used split should is in the name "test_classes.txt".

"/data/AWA2" --- The same as above, except that the raw data is in "/image" instead of "/feature". (In this submission raw images are deleted due to large size, can be downloaded at https://cvml.ist.ac.at/AwA2/AwA2-data.zip)


Reference:

[1] Wang, Wenlin, et al. "Zero-shot learning via class-conditioned deep generative models." arXiv preprint arXiv:1711.05820 (2017).

[2] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

[3] Y. Xian, C. H. Lampert, B. Schiele, Z. Akata. "Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly", IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI) 40(8), 2018. (arXiv:1707.00600 [cs.CV])
