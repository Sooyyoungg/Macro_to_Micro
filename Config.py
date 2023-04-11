class Config:
    ## dataset parameters
    train_list = '/scratch/connectome/conmaster/Projects/Image_Translation/DTI_process/train_subjects.csv'
    valid_list = '/scratch/connectome/conmaster/Projects/Image_Translation/DTI_process/val_subjects.csv'
    test_list = '/scratch/connectome/conmaster/Projects/Image_Translation/DTI_process/test_subjects.csv'

    data_num = 5000
    data = 'mri'
    t1_dir = '/storage/connectome/GANBERT/data/T1'
    dti_dir = '/storage/connectome/GANBERT/data/FA'

    # output directory
    #log_dir = '/pscratch/sd/s/sooyoung/Results_ffhq/log/moment'
    #ckpt_dir = '/pscratch/sd/s/sooyoung/Results_ffhq/ckpt/moment'
    #img_dir = '/pscratch/sd/s/sooyoung/Results_ffhq/Generated_images/moment'
    
    dir_n =  data
    file_n = dir_n + '_' + str(data_num)+'_FA_E256_Perceptloss'
    log_dir = './log/' + dir_n + '/' + file_n
    ckpt_dir = './ckpt/' + file_n
    img_dir = './Generated_images/' + file_n

    # VGG pre-trained model
    vgg_model = './vgg_normalised.pth'

    ## basic parameters
    n_epoch = 100
    n_iter = 100
    n_iter_decay = 100
    batch_size = 8
    lr = 0.0001
    lr_policy = 'step'
    lr_decay_iters = 50
    beta1 = 0.0
    pool_size = 50
    image_display_iter = 100
    gan_mode = 'swap'
   
    phase = 'train'
    train_continue = 'off'  # on / off

    # preprocess parameters
    crop_size = 512 # 512 + 256
    preprocess_crop_padding = None

    # model parameters
    load_size = 256      # 256 for vision data / 128 for MRI
    input_nc = 1         # of input image channel
    feat_nc = 1024        # of Encoder feature map channel
    output_nc = 1        # of output image channel

    alpha_in = 0.5
    alpha_out = 0.5
    freq_weight = 0
    real_freq = 0

    # Loss ratio
    lambda_img = 1.0
    lambda_pix = 1.0
    lambda_percept = 1.0
    lambda_GAN_G = 1.0
    lambda_GAN_D = 1.0
    lambda_PD_G = 1.0
    lambda_PD_D = 1.0
    lambda_trs = 1.0
    
    w_recon_pix_l1_low = 1.0
    w_recon_pix_moment_low = 0.0
    w_recon_pix_l1_high = 1.0
    w_recon_pix_moment_high = 1.0
    w_recon_fft_l1 = 1.0
    #w_recon_fft_moment = 1.0
    w_recon_fft_az = 1.0
    CMDweights = [1.0, 1.0, 1.0, 1.0, 1.0]  # len of weights = n-th moment loss
    
    w_trans_pix_l1 = 1.0
    w_trans_pix_moment = 1.0
    w_trans_fft_l1 = 1.0
    #w_trans_fft_moment = 1.0
    w_trans_fft_az = 1.0
    w_trans_PD_moment = 0.0 # for style
    w_az_loss_high = 0.0
    gauss_kernel_size = 21
    radius = 21

    ngf = 64
    ndf = 64
    # initial = True        # Initialize the Generator
    norm = 'instance'     # [instance | batch | none]
    init_type = 'normal'  # [normal | xavier | kaiming | orthogonal]
    init_gain = 0.02      # scaling factor for normal, xavier and orthogonal
    no_dropout = 'store_true'   # no dropout for generator
    
    patch_use_aggregation = True
    netPatchD_max_nc = 256+128
    netPatchD_scale_capacity = 4
    patch_min_scale = 1 / 8
    patch_max_scale = 1 / 4
    patch_num_crops = 8
    patch_size = 128     # half of the image size
    use_antialias = True

    # DDP configs:
    world_size = -1 # 'number of nodes for distributed training'
    rank = -1 # node rank for distributed training'
    dist_backend = 'nccl' # ='distributed backend'
    local_rank =-1 #'local rank for distributed training'
