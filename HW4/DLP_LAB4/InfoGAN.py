#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


class FrontEnd(nn.Module):
    def __init__(self):
        super(FrontEnd, self).__init__()
        self.main = nn.Sequential(
          nn.Conv2d(1, 64, 4, 2, 1),
#           nn.LeakyReLU(0.1, inplace=True),
          nn.ReLU(True),
          nn.Conv2d(64, 128, 4, 2, 1, bias=False),
          nn.BatchNorm2d(128),
#           nn.LeakyReLU(0.1, inplace=True),
          nn.ReLU(True),
          nn.Conv2d(128, 1024, 7, bias=False),
          nn.BatchNorm2d(1024),
#           nn.LeakyReLU(0.1, inplace=True),
          nn.ReLU(True),
        )

    def forward(self, x):
        output = self.main(x)
        return output


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
          nn.Conv2d(1024, 128, 1),
          nn.BatchNorm2d(128),
          nn.ReLU(True),  
          nn.Conv2d(128, 1, 1),
          nn.Sigmoid()
        )
    
    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output


class Q(nn.Module):
    def __init__(self):
        super(Q, self).__init__()
        self.conv = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn = nn.BatchNorm2d(128)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        y = self.conv(x)
        disc_logits = self.conv_disc(y).squeeze()
        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()
        return disc_logits, mu, var 

    
class G(nn.Module):
    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
          nn.ConvTranspose2d(64, 1024, 1, 1, bias=False),
          nn.BatchNorm2d(1024),
          nn.ReLU(True),
          nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
          nn.BatchNorm2d(128),
          nn.ReLU(True),
          nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
          nn.BatchNorm2d(64),
          nn.ReLU(True),
          nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
          nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# In[15]:


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - (x-mu).pow(2).div(var.mul(2.0)+1e-6)
        return logli.sum(1).mean().mul(-1)

class Trainer:
    def __init__(self, G, FE, D, Q):
        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q
        self.batch_size = 100

    def _noise_sample(self, dis_c, noise, bs):
        idx = np.random.randint(10, size=bs)
        c = np.zeros((bs, 10))
        c[range(bs),idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        noise.data.normal_()
        z = torch.cat([noise, dis_c], 1).view(-1, 64, 1, 1)
        return z, idx

    def train(self):
        rdloss, rgloss = [], []
        real_x = torch.FloatTensor(self.batch_size, 1, 28, 28).cuda()
        label = torch.FloatTensor(self.batch_size, 1).cuda()
        dis_c = torch.FloatTensor(self.batch_size, 10).cuda()
#         con_c = torch.FloatTensor(self.batch_size, 2).cuda()
        noise = torch.FloatTensor(self.batch_size, 54).cuda()

        real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
#         con_c = Variable(con_c)
        noise = Variable(noise)

        criterionD = nn.BCELoss().cuda()
        criterionQ_dis = nn.CrossEntropyLoss().cuda()
#         criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params':self.FE.parameters()}, {'params':self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
        optimG = optim.Adam([{'params':self.G.parameters()}, {'params':self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

        dataset = dset.MNIST('./dataset', transform=transforms.ToTensor(), download=True)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)

        # fixed random variables
        c = np.linspace(-1, 1, 10).reshape(1, -1)
        c = np.repeat(c, 10, 0).reshape(-1, 1)

        idx = np.arange(10).repeat(10)
        one_hot = np.zeros((100, 10))
        one_hot[range(100), idx] = 1
        fix_noise = torch.Tensor(100, 54).normal_()
        afix_noise = torch.Tensor(50, 54).normal_().double()
#         fix_noise = torch.Tensor(100, 54).uniform_(-1, 1)
#         afix_noise = torch.Tensor(50, 54).uniform_(-1, 1).double()

        for epoch in range(100):
            for num_iters, batch_data in enumerate(dataloader, 0):
                
                # real part
                optimD.zero_grad()

                x, _ = batch_data
                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs, 1)
                dis_c.data.resize_(bs, 10)
#                 con_c.data.resize_(bs, 2)
                noise.data.resize_(bs, 54)

                real_x.data.copy_(x)
                fe_out1 = self.FE(real_x)
                probs_real = self.D(fe_out1)
                label.data.fill_(1)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # fake part
                z, idx = self._noise_sample(dis_c, noise, bs)
#                 z, idx = self._noise_sample(dis_c, con_c, noise, bs)
                fake_x = self.G(z)
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()

                # G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                probs_fake = self.D(fe_out)
                label.data.fill_(1.0)

                reconstruct_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = self.Q(fe_out)
                class_ = torch.LongTensor(idx).cuda()
                target = Variable(class_)
                dis_loss = criterionQ_dis(q_logits, target)
#                 con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1

                G_loss = reconstruct_loss + dis_loss
#                 G_loss = reconstruct_loss + dis_loss + con_loss
                G_loss.backward()
                optimG.step()

                if num_iters % 100 == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(epoch, num_iters, D_loss.data.cpu().numpy(), G_loss.data.cpu().numpy()))
                    rdloss.append(D_loss.data.cpu().numpy())
                    rgloss.append(G_loss.data.cpu().numpy())
                    noise.data.copy_(fix_noise)
                    dis_c.data.copy_(torch.Tensor(one_hot))
#                     con_c.data.copy_(torch.from_numpy(c1))
                    z = torch.cat([noise, dis_c], 1).view(-1, 64, 1, 1)
#                     z = torch.cat([noise, dis_c, con_c], 1).view(-1, 64, 1, 1)
                    x_save = self.G(z)
                    save_image(x_save.data, './tmp/10x10.png', nrow=10)

            aidx = np.arange(5).repeat(10)
            aone_hot = np.zeros((50, 10))
            aone_hot[range(50), aidx] = 1
            z = torch.cat([afix_noise, torch.tensor(aone_hot)], 1).view(-1, 64, 1, 1)
#             z = torch.cat([afix_noise, torch.tensor(aone_hot), torch.tensor(ac1)], 1).view(-1, 64, 1, 1)
            save_image(self.G(z.float().cuda()), './tmp/test_{}.png'.format(str(epoch)), nrow=10)
        
        return rdloss, rgloss, self.G


# In[16]:


fe = FrontEnd()
d = D()
q = Q()
g = G()

for i in [fe, d, q, g]:
    i.cuda()
    i.apply(weights_init)

trainer = Trainer(g, fe, d, q)
dloss, gloss, generator = trainer.train()


# In[17]:


md, mg = np.array(dloss), np.array(gloss)
plt.figure(figsize=(8, 6))
plt.plot(range(md.shape[0]), md, label="Dloss")
plt.plot(range(mg.shape[0]), mg, label="Gloss")
plt.title("Generator and Discriminator Loss During Training")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('Loss_epoch_150.png')
plt.show()


# In[18]:


torch.save(generator.state_dict(), 'infogan_150.pkl')


# In[19]:


model = G()
model.load_state_dict(torch.load('infogan_150.pkl'))
model.eval()


# In[20]:


aidx = np.arange(5).repeat(10)
aone_hot = np.zeros((50, 10))
aone_hot[range(50), aidx] = 1
afix_noise = torch.Tensor(50, 54).normal_().double()


z = torch.cat([afix_noise, torch.tensor(aone_hot)], 1).view(-1, 64, 1, 1)
save_image(model(z.float()), './tmp/load_model150.png', nrow=10)


# In[21]:


aidx = np.arange(5).repeat(10)
aone_hot = np.zeros((50, 10))
aone_hot[range(50), aidx] = 1
afix_noise = torch.Tensor(50, 54).normal_().double()


z = torch.cat([afix_noise, torch.tensor(aone_hot)], 1).view(-1, 64, 1, 1)
save_image(trainer.G(z.float().cuda()), './tmp/no_model150.png', nrow=10)


# In[22]:


print(g)


# In[23]:


print(d)


# In[ ]:




