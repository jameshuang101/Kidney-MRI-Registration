import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"

# Paths
HOME_PATH = '/home/jmh170430/Documents/mnt1/Advisory_Folder/Data/SRM_DCE_20DataSet/'
TRAIN_PATH = os.path.join(HOME_PATH, 'Train_aff/')
VAL_PATH = os.path.join(HOME_PATH, 'Validate_aff/')
TEST_PATH = os.path.join(HOME_PATH, 'Test_aff/')
#WEIGHTS_PATH = os.path.join(HOME_PATH, 'Voxelmorph/brain_2d_smooth.h5')
WEIGHTS_PATH = os.path.join(HOME_PATH, 'Weights/Seg2D_Deform_Dice_1_19.h5')
HISTORY_PATH = os.path.join(HOME_PATH, 'History/Seg2D_Deform_Dice_1_19')
VXM_DATA_PATH = os.path.join(HOME_PATH, 'Voxelmorph/tutorial_data.npz')
DATA_PATH = os.path.join(HOME_PATH, 'data_cropped/')
DATA_PATH_NPY = os.path.join(HOME_PATH, 'data_npy/')
PRED_PATH = os.path.join(HOME_PATH, 'Predictions/')
PLOTS_PATH = os.path.join(HOME_PATH, 'Plots/')
IMAGES_PATH = os.path.join(HOME_PATH, 'Images/')
OUTPUTS_PATH = os.path.join(HOME_PATH, 'Outputs/')
VIDS_PATH = os.path.join(HOME_PATH, 'Videos/')
LABEL2D_PATH = os.path.join(HOME_PATH, 'segmentations2d_cropped/')
LABEL2D_TR_PATH = os.path.join(HOME_PATH, 'Train_seg/')
LABEL2D_TS_PATH = os.path.join(HOME_PATH, 'Test_seg/')
NII_PATH = os.path.join(HOME_PATH, 'Niftis/')
GIF_PATH = os.path.join(HOME_PATH, 'GIFs/')
EVAL_PATH = os.path.join(HOME_PATH, 'Eval_Outputs/')
OFRAMES_PATH = os.path.join(HOME_PATH, 'Outputs_Frames/')
EFRAMES_PATH = os.path.join(HOME_PATH, 'Eval_Outputs_Frames/')

# Training
BATCH_SIZE = 64
IMG_SIZE = 384
IMG_WIDTH = 384
IMG_HEIGHT = 224
NB_EPOCHS = 50
STEPS_PER_EPOCH = 25
VAL_STEPS = 4
PATIENCE = 10
LAMBDA_PARAM = 0.05
ALPHA_PARAM = 1
BETA_PARAM = 0.05
USING_PRETRAINED_WEIGHTS = False
TO_AUGMENT = False

# Predictions
PATIENT_NAME = 'Patient_004_Slice_013'
SLICE_NB = 10
TEST_BATCH_SIZE = 10
TO_EVAL = True
COMP_FRAME = 6
PEAK_FRAME = 6
