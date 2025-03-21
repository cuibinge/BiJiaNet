import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP 
import torch.distributed as dist
import torch.multiprocessing as mp 
import utils
from model import DiffusionNet

import os
import shutil
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import numpy as np
from scipy.io import savemat

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512
EPOCHS = 200
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
T = 1000
DATA_TYPE = 'diff_quadratic'

betas = utils.quadratic_beta_schedule(timesteps=T)
betas_schedule = utils.get_beta_schedule(betas)


def cleanup():
    pass


@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = utils.get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = utils.get_index_from_list(
        betas_schedule['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = utils.get_index_from_list(betas_schedule['sqrt_recip_alphas'], t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = utils.get_index_from_list(betas_schedule['posterior_variance'], t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 
        

@torch.no_grad()
def sample_plot_image(model, gpu, epoch):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=gpu)
    plt.figure()
    plt.axis('off')
    num_images = 100
    stepsize = int(T/num_images)

    all_images = []
    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=gpu, dtype=torch.long)
        img = sample_timestep(model, img, t)
        if i % stepsize == 0:
            all_images.append(img)

    fig, axs = plt.subplots(10,10)
    x=0
    for i in range(10):
        for j in range(10):
            out_img = utils.reverse_transforms_image(all_images[x].detach().cpu())
            axs[i,j].imshow(out_img)
            axs[i,j].axis('off')
            x += 1
    plt.savefig('./images/'+DATA_TYPE+'/image_' + str(epoch) + '.png', dpi=300) 


def initialize_weights(model):
    # Initializes weights according to the normal distribution
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.normal_(m.weight.data, 0.0, 0.01)


def train_epoch(train_dataloader, model, optimizer, gpu, epoch, args):

    model.train()
    losses = []
    p_bar = tqdm(train_dataloader)

    for img_batch in p_bar:
        optimizer.zero_grad()

        img_batch = img_batch.to(gpu, non_blocking=False)
        t = torch.randint(0, T, (img_batch.shape[0],)).long()
        t = t.to(gpu, non_blocking=False)
        x_noisy, noise = utils.forward_diffusion_sample(img_batch, t, betas_schedule, gpu)
        noise_pred = model(x_noisy, t)
        loss = utils.get_loss(noise, noise_pred, t, betas_schedule, gpu)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        p_bar.set_description('Epoch {}'.format(epoch))
        p_bar.set_postfix(loss=loss.item())

    print('Epoch: {}\ttotal_loss {:.4f}'.format(epoch, np.mean(losses)))   

    return np.mean(losses)


def eval_epoch(eval_dataloader, model, gpu, epoch, args, early_stopping=None):

    with torch.no_grad():
        model.eval()
        losses = []
        p_bar = tqdm(eval_dataloader)

        for img_batch in p_bar:
            img_batch = img_batch.to(gpu, non_blocking=False)
            t = torch.randint(0, T, (img_batch.shape[0],)).long()
            t = t.to(gpu, non_blocking=False)
            x_noisy, noise = utils.forward_diffusion_sample(img_batch, t, betas_schedule, gpu)
            noise_pred = model(x_noisy, t)
            loss = utils.get_loss(noise, noise_pred, t, betas_schedule, gpu)

            losses.append(loss.item())

            p_bar.set_description('Epoch {}'.format(epoch))
            p_bar.set_postfix(loss=loss.item())
    
    print('Epoch: {}\ttotal_loss {:.4f}'.format(epoch, np.mean(losses)))   

    return np.mean(losses)    


def main(gpu, args):
    torch.cuda.set_device(0)

    data_size = None

    # data loaders
    train_dataset, eval_dataset = utils.load_transformed_dataset()
                         
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                drop_last=True, num_workers=4, pin_memory=True,
                                shuffle=True)
                             
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, drop_last=False, num_workers=4, pin_memory=True)                            

    model = DiffusionNet(dim=64, channels=3).to(gpu)
    initialize_weights(model)
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = args['checkpoints_path']
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs('./images/' + DATA_TYPE, exist_ok=True)

    if args['load_from_chkpt'] is not None:
        chkpt_file = args['load_from_chkpt']
        print('Loading checkpoint from:', chkpt_file)
        checkpoint = torch.load(chkpt_file,map_location='cuda:0')
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
    
    epoch_start = 0

    early_stopping = utils.EarlyStopping(patience=15, verbose=True,
                                             path=checkpoint_path + '{}_{}.pth'.format(BATCH_SIZE, LEARNING_RATE))
    

    train_losses = []
    eval_losses = []
    start_time = time.process_time()
    for epoch in range(epoch_start, EPOCHS):
        print('epoch {}/{}'.format(epoch + 1, EPOCHS))
        # train_sampler.set_epoch(epoch)
        train_loss = train_epoch(train_dataloader, model, optimizer, gpu, epoch + 1, args)
        eval_loss = eval_epoch(eval_dataloader, model, gpu, epoch + 1, early_stopping)

        mean_train_loss = torch.tensor(train_loss / args['gpus']).to(gpu)
        mean_eval_loss = torch.tensor(eval_loss / args['gpus']).to(gpu)

        # dist.barrier()
        # dist.all_reduce(mean_train_loss)
        # dist.all_reduce(mean_eval_loss)
        print('gpu {} eval_loss:{}, mean_loss:{}'.format(gpu, eval_loss,
                                                         mean_eval_loss.cpu().numpy()))

        # if optim_name.split('-')[-1] == 'step':
        #     scheduler.step(mean_eval_loss.cpu().numpy())
        # elif optim_name.split('-')[-1] == 'cosine':
        #     scheduler.step()
        # elif optim_name.split('-')[-1] == 'no':
        #     pass

        if (epoch+1) % 20 == 0:
            sample_plot_image(model, gpu, epoch+1)
        
        if gpu == 0:
            early_stopping(mean_eval_loss.cpu().numpy(), model, epoch + 1, ddp=False)
        
        train_losses.append(mean_train_loss.cpu().numpy())
        eval_losses.append(mean_eval_loss.cpu().numpy())

    current_time = time.process_time()
    print("Total Time Elapsed={:12.5} seconds".format(str(current_time - start_time)))

    # saving the plots
    plots_path = './plots/diff'
    os.makedirs(plots_path, exist_ok=True)
    epochs = np.arange(EPOCHS)
    train_losses = np.array(train_losses)
    eval_losses = np.array(eval_losses)
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    axes.plot(epochs, train_losses, 'tab:blue', epochs, eval_losses, 'tab:orange')
    axes.set_title(f'Training and Validation Loss (pretrained model = None, loss = MSE Loss, '
                   f'data size = {data_size})',
                   weight='bold', fontsize=7)
    axes.set_xlabel('Epochs', weight='bold', fontsize=9)
    axes.set_ylabel('Loss', weight='bold', fontsize=9)
    plt.savefig(plots_path + '/'+DATA_TYPE+'loss_' + str(data_size) + '.png', dpi=300)

    # os.makedirs('./loss', exist_ok=True)
    # path = os.path.join('./loss', DATA_TYPE+'_train_loss.mat')
    # mdic = {"data": train_losses, "label": "epochs"}
    # savemat(path, mdic)

    cleanup()


if __name__ == '__main__':
    
    args = {
        'gpus': 1, 
        'world_size': 1,  # 强制设置为单卡
        'checkpoints_path': './snapshots/' + DATA_TYPE + '/',
        'load_from_chkpt': None
    }
    main(gpu=0, args=args)  # 显式传递gpu=0

    print(args['gpus'])
