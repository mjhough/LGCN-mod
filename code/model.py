"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np

import pdb


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN2(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset, checkpt):
        super(LightGCN2, self).__init__()
        self.config = config
        self.checkpt = checkpt
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.mini_latent_dim = self.config['mini_latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        
        # Full model params
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.bias_user = nn.Embedding(self.num_users, 1)
        self.bias_item = nn.Embedding(self.num_items, 1)
        self.bias_user.weight.data.fill_(0.)
        self.bias_item.weight.data.fill_(0.)

        # Mini model params
        self.mm_embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.mini_latent_dim)
        self.mm_embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.mini_latent_dim)

        # proj mini embeddings to the full embeddings space
        self.proj = nn.Upsample(size=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.normal_(self.mm_embedding_user.weight, std=0.1)
        nn.init.normal_(self.mm_embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')

        # Load state_dict for just the full model:
        self.embedding_user.weight.data.copy_(self.checkpt['embedding_user.weight'])
        self.embedding_item.weight.data.copy_(self.checkpt['embedding_item.weight'])
        self.bias_user.weight.data.copy_(self.checkpt['bias_user.weight'])
        self.bias_item.weight.data.copy_(self.checkpt['bias_item.weight'])

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        all_emb = torch.cat([users_emb, items_emb])
        user_bias_emb = self.bias_user.weight
        item_bias_emb = self.bias_item.weight
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items, user_bias_emb, item_bias_emb

    def mm_computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.mm_embedding_user.weight
        items_emb = self.mm_embedding_item.weight

        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items, all_users_bias, all_items_bias = self.computer()
        mm_all_users, mm_all_items = self.mm_computer()

        users_emb = all_users[users.long()]
        mm_users_emb = mm_all_users[users.long()]
        items_emb = all_items
        mm_items_emb = mm_all_items
        users_bias_emb = all_users_bias[users.long()]
        items_bias_emb = all_items_bias

        # Project mm_users and mm_items into full space
        #proj_users_emb = self.proj_user(mm_users_emb)
        #proj_items_emb = self.proj_item(mm_items_emb)

        # Add embeddings together
        users_emb += self.proj(mm_users_emb.expand(1,-1,-1)).squeeze()
        items_emb += self.proj(mm_items_emb.expand(1,-1,-1)).squeeze()

        rating = torch.matmul(users_emb, items_emb.t()) # NxM matrix where N=users,M=items
        rating += users_bias_emb # Nx1 vector => adds column-wise for each user row
        rating += items_bias_emb.t() # 1xM vector => adds row-wise for each item column
        rating = self.f(rating)
        # rating = sigmoid(U @ V + B), where U is the users embedding matrix, V items, and B is the bias
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, all_users_bias, all_items_bias = self.computer()
        mm_all_users, mm_all_items = self.mm_computer()

        users_emb = all_users[users]
        mm_users_emb = mm_all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        mm_pos_emb = mm_all_items[pos_items]
        mm_neg_emb = mm_all_items[neg_items]

        # Add embeddings together
        users_emb += self.proj(mm_users_emb.expand(1,-1,-1)).squeeze()
        pos_emb += self.proj(mm_pos_emb.expand(1,-1,-1)).squeeze()
        neg_emb += self.proj(mm_neg_emb.expand(1,-1,-1)).squeeze()
        
        users_bias_emb = all_users_bias[users]
        pos_bias_emb = all_items_bias[pos_items]
        neg_bias_emb = all_items_bias[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        #mm_users_emb_ego = self.mm_embedding_user(users)
        #mm_pos_emb_ego = self.mm_embedding_item(pos_items)
        #mm_neg_emb_ego = self.mm_embedding_item(neg_items)

        # Add embeddings together
        #users_emb_ego += self.proj(mm_users_emb_ego.expand(1,-1,-1)).squeeze()
        #pos_emb_ego += self.proj(mm_pos_emb_ego.expand(1,-1,-1)).squeeze()
        #neg_emb_ego += self.proj(mm_neg_emb_ego.expand(1,-1,-1)).squeeze()

        return users_emb, pos_emb, neg_emb, users_bias_emb, pos_bias_emb, neg_bias_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, users_bias_emb, pos_bias_emb, neg_bias_emb,
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        # userBiasEmb0, posBiasEmb0, negBiasEmb0) 
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb) + (users_bias_emb + pos_bias_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb) + (users_bias_emb + neg_bias_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.bias_user = nn.Embedding(self.num_users, 1)
        self.bias_item = nn.Embedding(self.num_items, 1)
        self.bias_user.weight.data.fill_(0.)
        self.bias_item.weight.data.fill_(0.)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        user_bias_emb = self.bias_user.weight
        item_bias_emb = self.bias_item.weight
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items, user_bias_emb, item_bias_emb
    
    def getUsersRating(self, users):
        all_users, all_items, all_users_bias, all_items_bias = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        users_bias_emb = all_users_bias[users.long()]
        items_bias_emb = all_items_bias

        rating = torch.matmul(users_emb, items_emb.t()) # NxM matrix where N=users,M=items
        rating += users_bias_emb # Nx1 vector => adds column-wise for each user row
        rating += items_bias_emb.t() # 1xM vector => adds row-wise for each item column
        rating = self.f(rating)
        # rating = sigmoid(U @ V + B), where U is the users embedding matrix, V items, and B is the bias
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items, all_users_bias, all_items_bias = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_bias_emb = all_users_bias[users]
        pos_bias_emb = all_items_bias[pos_items]
        neg_bias_emb = all_items_bias[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        #users_bias_emb_ego = self.bias_user(users)
        #pos_bias_emb_ego = self.bias_item(pos_items)
        #neg_bias_emb_ego = self.bias_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_bias_emb, pos_bias_emb, neg_bias_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, users_bias_emb, pos_bias_emb, neg_bias_emb,
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        # userBiasEmb0, posBiasEmb0, negBiasEmb0) 
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb) + (users_bias_emb + pos_bias_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb) + (users_bias_emb + neg_bias_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items, all_users_bias, all_items_bias = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]

        users_bias_emb = all_users_bias[users]
        items_bias_emb = all_items_bias[items]

        inner_pro = users_bias_emb + items_bias_emb
        inner_pro += torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
