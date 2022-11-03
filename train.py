import cv2
import os
import glob
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from datetime import datetime
import imageio
import torchvision.utils as utils

from loader import *
from utils import *
from model import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path = './data/train_data'


global params
params = {'path': path,
          'batch_size': 4,
          'output_size': 256,
          'gf_dim': 32,
          'df_dim': 32,
          'model_path': 'model',
          'L1_lambda': 100,
          'lr': 0.0001,
          'beta_1': [0.5, 0.999],
          'epochs': 50,
          'Img_saved_path': 'Saved/SavedImgs',
          'Img_saved_path_for_real_data': 'Saved/SavedReal',
          'Stage_epochs': [10000, 20000]}


if not os.path.isdir(params['Img_saved_path']):
    os.mkdir(params['Img_saved_path'])

if not os.path.isdir(params['Img_saved_path_for_real_data']):
    os.mkdir(params['Img_saved_path_for_real_data'])


def get_file_paths(path):
    img_path = [os.path.join(root,file) for root, dirs, files in os.walk(path) for file in files if 'GT' not in file]
    gt_path = [os.path.join(os.path.dirname(file), os.path.basename(file).split('.')[0] + '_GT.png') for file in img_path]
    return np.array(img_path[:int(len(img_path)*.9)]), np.array(gt_path[:int(len(gt_path)*.9)]), np.array(img_path[int(len(img_path)*.9):]) , np.array(gt_path[int(len(gt_path)*.9):])


def load_data_CONTENT(path):
    im = ~cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(256,256))
    return np.expand_dims(im, -1)/127.5 - 1.

def load_data(path):
    im = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE),(256,256))
    return np.expand_dims(im, -1)/127.5 - 1.


# 추가 (
content = torch.randn(params['batch_size'],1,256,256)
content.to(device)
style = torch.randn(params['batch_size'],1,256,256)
style.to(device)
real_input = torch.randn(params['batch_size'],1,256,256)
real_input.to(device)
#Real = torch.randn(params['batch_size'],1,256,256)



tg_output = Generator1(content.size(), style.size())
tg_output = tg_output.to(device)
bg_cleaned= Generator2(content.size())
bg_cleaned = bg_cleaned.to(device)



discriminator = Discriminator1(style.size())
discriminator = discriminator.to(device)
discriminator2 = Discriminator2(content.size())
discriminator2 = discriminator2.to(device)


# optimizer
d1_optim = torch.optim.Adam(discriminator.parameters(), lr=params['lr'], betas=params['beta_1'])
g1_optim = torch.optim.Adam(tg_output.parameters(), lr=params['lr'], betas=params['beta_1'])

d2_optim = torch.optim.Adam(discriminator2.parameters(), lr=params['lr'], betas=params['beta_1'])
g2_optim = torch.optim.Adam(bg_cleaned.parameters(), lr=params['lr'], betas=params['beta_1'])
# 추가 )

train_STYLEpath, train_CONTENTpath, test_STYLEpath, test_CONTENTpath = get_file_paths(params['path'])

np.random.shuffle(train_STYLEpath)
np.random.shuffle(train_CONTENTpath)

data_set_size = len(train_STYLEpath)
#print(data_set_size//params['batch_size'])

rndm_test_style = test_STYLEpath[np.random.choice(len(test_STYLEpath),params['batch_size']*10, replace = True)]
rndm_test_content = test_CONTENTpath[np.random.choice(len(test_CONTENTpath),params['batch_size']*10, replace = True)]

Fixd_rndm_indx = np.random.choice(len(test_STYLEpath),params['batch_size']*10, replace = True)

paired_smaple_input = test_STYLEpath[Fixd_rndm_indx]
paired_smaple_output = test_CONTENTpath[Fixd_rndm_indx]

start_time = time.time()
counter =1

losses = []
for epoch in range(params['epochs']):
    #print("Epoch:{}".format(epoch))
    for idx in range(data_set_size//params['batch_size']):
        batch_STYLEpath = train_STYLEpath[idx * params['batch_size']: (idx + 1) * params['batch_size']]
        batch_STYLEdata = torch.Tensor(np.array([load_data(path) for path in batch_STYLEpath]))
        batch_CONTENTpath = train_CONTENTpath[idx * params['batch_size']: (idx + 1) * params['batch_size']]
        batch_CONTENTdata = torch.Tensor(np.array([load_data_CONTENT(path) for path in batch_CONTENTpath]))

        content_input = batch_CONTENTdata
        style_input = batch_STYLEdata
        content_input = torch.permute(content_input, (0, 3, 1, 2))
        style_input = torch.permute(style_input, (0, 3, 1, 2))
        content_input = content_input.to(device)
        style_input = style_input.to(device)

        # TANet train
        if counter<=params['Stage_epochs'][0]:


            # content_input = (1, params['output_size'], params['output_size'])
            # style_input = (3, params['output_size'], params['output_size'])


           # g_output.apply(weights_init_normal)
            g1_output = tg_output(content_input, style_input)

           # loss
            # Discriminator Train
            D1_real, D1_real_logits = discriminator(style_input)
            D1_fake, D1_fake_logits = discriminator(g1_output)
            d1_loss_real = torch.mean(F.binary_cross_entropy_with_logits(D1_real_logits, torch.ones_like(D1_real)))
            d1_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(D1_fake_logits, torch.zeros_like(D1_fake)))
            d1_loss = d1_loss_fake + d1_loss_real

            d1_optim.zero_grad()
            d1_loss.backward(retain_graph=True)


            content_losses = content_loss(g1_output, content_input)
            style_losses = style_loss(g1_output, style_input)
            g1_loss = torch.mean(F.binary_cross_entropy_with_logits(D1_fake_logits, torch.ones_like(D1_fake)))
            g1_loss = g1_loss + 10 * content_losses + 0.5 * style_losses
            g1_loss.to(device)

            # Generator Train
            g1_optim.zero_grad()
            g1_loss.backward()
            d1_optim.step()
            g1_optim.step()


            losses.append([g1_loss.item(), d1_loss.item()])
            if counter%10==0:
                print("# stage 1 : [Epoch %d/%d] [Batch %d] [D1 loss: %f] [G1 loss: %f]" % (epoch, params['epochs'], idx, d1_loss.item(), g1_loss.item()))

        elif counter>params['Stage_epochs'][0] and counter<=params['Stage_epochs'][1]:

            g1_output = tg_output(content_input, style_input)
            # binet train
            g2_output = bg_cleaned(g1_output)
            # Discriminator Train
            D2_real, D2_real_logits = discriminator2(content_input)
            D2_fake, D2_fake_logits = discriminator2(g2_output)
            d2_loss_real = torch.mean(F.binary_cross_entropy_with_logits(D2_real_logits, torch.ones_like(D2_real)))
            d2_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(D2_fake_logits, torch.zeros_like(D2_fake)))
            d2_loss = d2_loss_fake + d2_loss_real

            d2_optim.zero_grad()
            d2_loss.backward(retain_graph=True)

            g2_loss = torch.mean(F.binary_cross_entropy_with_logits(D2_fake_logits, torch.ones_like(D2_fake)))
            g2_loss = g2_loss + 100 * torch.mean(torch.abs(content_input - g2_output))
            g2_loss.to(device)

            # Generator Train
            g2_optim.zero_grad()
            g2_loss.backward()
            d2_optim.step()
            g2_optim.step()

            if counter%10==0:
                print("# stage 2 : [Epoch %d/%d] [Batch %d] [D2 loss: %f] [G2 loss: %f]" % (epoch, params['epochs'], idx, d2_loss.item(), g2_loss.item()))

        else:
            # tanet, binet combined train
            g1_output_ = tg_output(content_input, style_input)

            # loss
            # Discriminator Train
            D1_real, D1_real_logits = discriminator(style_input)
            D1_fake, D1_fake_logits = discriminator(g1_output_)
            d1_loss_real = torch.mean(F.binary_cross_entropy_with_logits(D1_real_logits, torch.ones_like(D1_real)))
            d1_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(D1_fake_logits, torch.zeros_like(D1_fake)))
            d1_loss_ = d1_loss_fake + d1_loss_real

            d1_optim.zero_grad()
            d1_loss_.backward(retain_graph=True)

            content_losses_ = content_loss(g1_output_, content_input)
            style_losses_ = style_loss(g1_output_, style_input)
            g1_loss_ = torch.mean(F.binary_cross_entropy_with_logits(D1_fake_logits, torch.ones_like(D1_fake)))
            g1_loss_ = g1_loss_ + 10 * content_losses_ + 0.5 * style_losses_
            g1_loss_.to(device)

            # Generator Train
            g1_optim.zero_grad()
            g1_loss_.backward(retain_graph=True)
            d1_optim.step()
            g1_optim.step()


            # binet train
            g2_output_ = bg_cleaned(g1_output_)
            # Discriminator Train
            D2_real, D2_real_logits = discriminator2(content_input)
            D2_fake, D2_fake_logits = discriminator2(g2_output_)
            d2_loss_real = torch.mean(F.binary_cross_entropy_with_logits(D2_real_logits, torch.ones_like(D2_real)))
            d2_loss_fake = torch.mean(F.binary_cross_entropy_with_logits(D2_fake_logits, torch.zeros_like(D2_fake)))
            d2_loss_ = d2_loss_fake + d2_loss_real

            d2_optim.zero_grad()
            d2_loss_.backward(retain_graph=True)

            g2_loss_ = torch.mean(F.binary_cross_entropy_with_logits(D2_fake_logits, torch.ones_like(D2_fake)))
            g2_loss_ = g2_loss_ + 100 * torch.mean(torch.abs(content_input - g2_output_))
            g2_loss_.to(device)

            # Generator Train
            g2_optim.zero_grad()
            g2_loss_.backward()
            d2_optim.step()
            g2_optim.step()
            if counter%10==0:
                print("# Combined stage : [Epoch %d/%d] [Batch %d] [D1 loss: %f] [G1 loss: %f] [D2 loss: %f] [G2 loss: %f]" % (epoch, params['epochs'], idx, d1_loss_.item(), g1_loss_.item() , d2_loss_.item(), g2_loss_.item()))

        counter = counter + 1
        if counter % 20 == 0:
            save_image(g2_output_.data[:5], params['Img_saved_path_for_real_data'] + '/train_3/2-{}.png'.format(epoch), nrow=5, normalize=True)
        # print("# stage 2 : [Epoch %d/%d] [Batch %d] [D2 loss: %f] [G2 loss: %f]" % (epoch, params['epochs'], idx, d2_loss.item(), g2_loss.item()))

        #save_image(g1_output_.data[:5],params['Img_saved_path']+'/train_3/2-{}.png'.format(epoch), nrow = 5, normalize = True)

            torch.save(tg_output.state_dict(), params['model_path']+"/"+'G_params.pt')
            torch.save(discriminator.state_dict(), params['model_path']+"/"+'D_params.pt')

    print('Model Saved!!')
       # g1_output = torch.permute(g1_output,(0,2,3,1))
       # g1_output = g1_output.to('cpu')
       # g1_output = g1_output.detach().numpy()

'''
        elif counter % 500 == 0:

            if not os.path.isdir(os.path.join(params['Img_saved_path'], str(counter))):
                os.mkdir(os.path.join(params['Img_saved_path'], str(counter)))

            if not os.path.isdir(os.path.join(params['Img_saved_path_for_real_data'], str(counter))):
                os.mkdir(os.path.join(params['Img_saved_path_for_real_data'], str(counter)))

            cc = 1

            for k in range(10):

                batchx = paired_smaple_input[k * params['batch_size']: (k + 1) * params['batch_size']]
                batchx_data = np.array([load_data(path) for path in batchx])

                batchy = paired_smaple_output[k * params['batch_size']: (k + 1) * params['batch_size']]
                batchy_data = np.array([load_data_CONTENT(path) for path in batchy])

                #feed_dict = {style_input: batchx_data}
                #batchx_data[1],batchx_data[2],batchx_data[3] = batchx_data[2],batchx_data[3],batchx_data[1]
                #batchy_data[1],batchy_data[2],batchy_data[3] = batchy_data[2],batchy_data[3],batchy_data[1]

                #batchx_data = batchx_data.cpu()
                #batchy_data = batchy_data.cpu()

                all_imgs = np.concatenate((batchx_data, batchy_data), 2)

                for no_i in range(params['batch_size']):
                    imageio.imwrite(
                        os.path.join(params['Img_saved_path_for_real_data'], str(counter), str(cc) + ".jpg"),
                        all_imgs[no_i, :, :, :])

                    cc = cc + 1

            cc = 1

            for k in range(10):

                batch_STYLEpath2 = rndm_test_style[k * params['batch_size']: (k + 1) * params['batch_size']]
                batch_STYLEdata2 = torch.Tensor([load_data(path) for path in batch_STYLEpath])
                batch_CONTENTpath2 = rndm_test_content[k * params['batch_size']: (k + 1) * params['batch_size']]
                batch_CONTENTdata2 = torch.Tensor([load_data_CONTENT(path) for path in batch_CONTENTpath])

                #feed_dict = {content_input: batch_CONTENTdata, style_input: batch_STYLEdata}
                batch_CONTENTdata2 = torch.permute(batch_CONTENTdata2, (0, 3, 1, 2)).to(device)
                batch_STYLEdata2 = torch.permute(batch_STYLEdata2, (0, 3, 1, 2)).to(device)
                g2_output = Generator1(batch_CONTENTdata2.size(), batch_STYLEdata2.size())
                g2_output = g2_output.to(device)
               # g1_output = torch.permute(g1_output,(0,2,3,1))
                g3_output = g2_output(batch_CONTENTdata2, batch_STYLEdata2)


               # g1_output = g1_output.swapaxes(1,2)
               # g1_output = g1_output.swapaxes(2,3)
                #g1_output = np.reshape(g1_output,(4,256,256,1))


                #all_imgs = np.concatenate((batch_CONTENTdata, batch_STYLEdata, g1_output), 2)

                all_imgs2 = torch.cat((batch_CONTENTdata2, batch_STYLEdata2, g3_output), 2)
              #  all_imgs2 = torch.permute(all_imgs2, (0, 3, 1, 2))
                all_imgs3 = all_imgs2.cpu().detach().numpy()

                #all_imgs3 = all_imgs2.detach().numpy()

                for no_i in range(params['batch_size']):
                    all_imgs2 = all_imgs2*255
               #     imageio.imwrite(os.path.join(params['Img_saved_path'], str(counter), str(cc) + ".jpg"), all_imgs3[no_i,:,:,:])
                    save_image(all_imgs2[no_i,:,:,:],os.path.join(params['Img_saved_path'], str(counter), str(cc) + ".jpg"), nrow=5, normalize=True)
                    cc = cc + 1
'''
