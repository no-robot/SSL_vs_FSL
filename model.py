from backbone import *
from lemniscate.lib.NCEAverage import NCEAverage
from lemniscate.lib.NCECriterion import NCECriterion


class AE(BaseClass):
    """Regular Autoencoder"""
    def __init__(self, backbone, in_channels, image_size, z_dim):
        super(AE, self).__init__()
        add_fc = True if image_size >= 32 else False  # add fully connected layer on top for larger images
        last_relu = False  # no ReLU in final layer
        self.encoder = Encoder(backbone, in_channels, image_size, z_dim, add_fc=add_fc, last_relu=last_relu)
        self.decoder = Decoder(backbone, z_dim, in_channels, image_size)

    def encode(self, x, last_layer=None):
        return self.encoder(x, last_layer)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, **kwargs):
        x = kwargs['img']
        z = self.encode(x).flatten(start_dim=1)
        x_recon = self.decode(z)
        out = {'x_recon': x_recon}
        return out

    @staticmethod
    def update(optim, **kwargs):
        x = kwargs['img']
        output = kwargs['output']
        x_recon = output['x_recon']
        loss = F.mse_loss(x_recon, x)

        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss


class VAE(BaseClass):
    """Variational Autoencoder (Kingma, D.P., Welling, M.: Auto-encoding variational bayes)"""
    def __init__(self, backbone, in_channels, image_size, z_dim):
        super(VAE, self).__init__()
        add_fc = True if image_size >= 32 else False  # add fully connected layer on top for larger images
        last_relu = False  # no ReLU in final layer
        self.z_dim = z_dim
        self.encoder = Encoder(backbone, in_channels, image_size, 2 * z_dim, add_fc=add_fc, last_relu=last_relu)
        self.decoder = Decoder(backbone, z_dim, in_channels, image_size)

    def encode(self, x, last_layer=None):
        out = self.encoder(x, last_layer)
        # restrict output of last encoder layer to latent mean
        if last_layer == self.encoder.num_layers:
            out = out[:, :self.z_dim]
        return out

    def decode(self, x):
        return self.decoder(x)

    def forward(self, **kwargs):
        x = kwargs['img']
        z = self.encode(x).flatten(start_dim=1)
        # sample latent variable
        mu = z[:, :self.z_dim]
        logvar = z[:, self.z_dim:]
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        z = mu + std * eps
        # reconstruct output
        x_recon = self.decode(z)
        out = {'x_recon': x_recon, 'mu': mu, 'logvar': logvar}
        return out

    def update(self, optim, **kwargs):
        x = kwargs['img']
        output = kwargs['output']
        x_recon = output['x_recon']
        mu = output['mu']
        logvar = output['logvar']
        batch_size = mu.size(0)

        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
        kld = (0.5 * (logvar.exp() + mu.pow(2) - logvar - 1)).sum(1).mean()
        loss = recon_loss + kld

        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss


class RotNet(BaseClass):
    """This code is modified from https://github.com/gidariss/FeatureLearningRotNet"""
    def __init__(self, backbone, in_channels, image_size, z_dim):
        super(RotNet, self).__init__()
        add_fc = True if image_size >= 32 else False  # add fully connected layer on top for larger images
        self.encoder = Encoder(backbone, in_channels, image_size, z_dim, add_fc=add_fc)
        self.fc = nn.Linear(z_dim, 4)

    def encode(self, x, last_layer=None):
        return self.encoder(x, last_layer)

    def decode(self, x):
        out = self.fc(x)
        return out

    def forward(self, **kwargs):
        x = kwargs['img']
        x_rot = torch.cat([x.rot90(l, [2, 3]) for l in range(4)])
        z = self.encode(x_rot).flatten(start_dim=1)
        scores = self.decode(z)
        target = torch.cat([torch.LongTensor([i]).repeat(x.size(0)) for i in range(4)]).to(x.device)
        out = {'scores': scores, 'target': target}
        return out

    @staticmethod
    def update(optim, **kwargs):
        output = kwargs['output']
        loss = F.cross_entropy(output['scores'], output['target'])

        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss


class AET(BaseClass):
    """This code is modified from https://github.com/maple-research-lab/AET"""
    def __init__(self, backbone, in_channels, image_size, z_dim):
        super(AET, self).__init__()
        add_fc = True if image_size >= 32 else False  # add fully connected layer on top for larger images
        self.encoder = Encoder(backbone, in_channels, image_size, z_dim, add_fc=add_fc)
        self.fc = nn.Linear(2 * z_dim, 8)

    def encode(self, x, last_layer=None):
        return self.encoder(x, last_layer)

    def decode(self, x):
        out = self.fc(x)
        return out

    def forward(self, **kwargs):
        x1 = kwargs['img']
        x2 = kwargs['img2']
        z1 = self.encode(x1).flatten(start_dim=1)
        z2 = self.encode(x2).flatten(start_dim=1)
        z = torch.cat((z1, z2), 1)
        out = self.decode(z)
        return out

    @staticmethod
    def update(optim, **kwargs):
        output = kwargs['output']
        loss = F.mse_loss(output, kwargs['p'])
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss


class NPID(BaseClass):
    """Non-parametric Instance Discrimination
    This code is modified from https://github.com/zhirongw/lemniscate.pytorch"""
    def __init__(self, backbone, in_channels, image_size, z_dim, n_data):
        super(NPID, self).__init__()
        nce_k = 4096
        nce_t = 0.2
        nce_m = 0.5
        add_fc = True if image_size >= 32 else False  # add fully connected layer on top for larger images
        last_relu = False  # no ReLU in final layer
        self.encoder = Encoder(backbone, in_channels, image_size, z_dim, add_fc=add_fc, last_relu=last_relu)
        self.lemniscate = NCEAverage(z_dim, n_data, nce_k, nce_t, nce_m)
        self.criterion = NCECriterion(n_data)

    def encode(self, x, last_layer=None):
        return self.encoder(x, last_layer)

    def forward(self, **kwargs):
        x = kwargs['img']
        out = self.encode(x).flatten(start_dim=1)
        out = F.normalize(out, p=2, dim=1)
        return out

    def update(self, optim, **kwargs):
        feature = kwargs['output']
        index = kwargs['idx']
        output = self.lemniscate(feature, index)
        loss = self.criterion(output, index)
        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss

    # Comment out the following method to train a linear classifier on the downstream tasks
    @staticmethod
    def train_classifier(z, n_way, n_support, n_query):
        # Assign classes according to cosine similarity to prototypes (normalized mean of support examples per class)
        z = F.normalize(z.view(n_way * (n_support + n_query), -1), p=2, dim=1)
        z = z.view(n_way, n_support + n_query, -1)

        z_support = z[:, :n_support]
        z_query = z[:, n_support:]

        z_proto = z_support.contiguous().view(n_way, n_support, -1).mean(1)
        z_proto = F.normalize(z_proto, p=2, dim=1)
        z_query = z_query.contiguous().view(n_way * n_query, -1)

        scores = torch.mm(z_query, z_proto.t())
        return scores


class ISIF(BaseClass):
    """Invariant and Spreading Instance Feature
    This code is modified from https://github.com/mangye16/Unsupervised_Embedding_Learning"""
    def __init__(self, backbone, in_channels, image_size, z_dim):
        super(ISIF, self).__init__()
        self.negM = 1
        self.T = 0.2
        add_fc = True if image_size >= 32 else False  # add fully connected layer on top for larger images
        last_relu = False  # no ReLU in final layer
        self.encoder = Encoder(backbone, in_channels, image_size, z_dim, add_fc=add_fc, last_relu=last_relu)

    def encode(self, x, last_layer=None):
        return self.encoder(x, last_layer)

    def forward(self, **kwargs):
        x = kwargs['img']
        z = self.encode(x).flatten(start_dim=1)
        z = F.normalize(z, p=2, dim=1)

        x2 = kwargs['img2']
        z2 = self.encode(x2).flatten(start_dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        out = torch.cat([z, z2], dim=0)
        return out

    def update(self, optim, **kwargs):
        x = kwargs['output']

        batchSize = x.size(0)
        diag_mat = 1 - torch.eye(batchSize).to(x.device)

        # get positive innerproduct
        reordered_x = torch.cat((x.narrow(0, batchSize // 2, batchSize // 2),
                                 x.narrow(0, 0, batchSize // 2)), 0)
        # reordered_x = reordered_x.data
        pos = (x * reordered_x.data).sum(1).div_(self.T).exp_()

        # get all innerproduct, remove diag
        all_prob = torch.mm(x, x.t().data).div_(self.T).exp_() * diag_mat
        if self.negM == 1:
            all_div = all_prob.sum(1)
        else:
            # remove pos for neg
            all_div = (all_prob.sum(1) - pos) * self.negM + pos

        lnPmt = torch.div(pos, all_div)

        # negative probability
        Pon_div = all_div.repeat(batchSize, 1)
        lnPon = torch.div(all_prob, Pon_div.t())
        lnPon = -lnPon.add(-1)

        # equation 7 in ref. A (NCE paper)
        lnPon.log_()
        # also remove the pos term
        lnPon = lnPon.sum(1) - (-lnPmt.add(-1)).log_()
        lnPmt.log_()

        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.sum(0)

        # negative multiply m
        lnPonsum = lnPonsum * self.negM
        loss = - (lnPmtsum + lnPonsum) / batchSize

        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss

    # Comment out the following method to train a linear classifier on the downstream tasks
    @staticmethod
    def train_classifier(z, n_way, n_support, n_query):
        # Assign classes according to cosine similarity to prototypes (normalized mean of support examples per class)
        z = F.normalize(z.view(n_way * (n_support + n_query), -1), p=2, dim=1)
        z = z.view(n_way, n_support + n_query, -1)

        z_support = z[:, :n_support]
        z_query = z[:, n_support:]

        z_proto = z_support.contiguous().view(n_way, n_support, -1).mean(1)
        z_proto = F.normalize(z_proto, p=2, dim=1)
        z_query = z_query.contiguous().view(n_way * n_query, -1)

        scores = torch.mm(z_query, z_proto.t())
        return scores


class ISIF_M(BaseClass):
    """Modified version of the ISIF model"""
    def __init__(self, backbone, in_channels, image_size, z_dim):
        super(ISIF_M, self).__init__()
        self.tau = 0.2
        add_fc = True if image_size >= 32 else False  # add fully connected layer on top for larger images
        last_relu = False  # no ReLU in final layer
        self.encoder = Encoder(backbone, in_channels, image_size, z_dim, add_fc=add_fc, last_relu=last_relu)

    def encode(self, x, last_layer=None):
        return self.encoder(x, last_layer)

    def forward(self, **kwargs):
        x = kwargs['img']
        z = self.encode(x).flatten(start_dim=1)
        z = F.normalize(z, p=2, dim=1)

        x2 = kwargs['img2']
        z2 = self.encode(x2).flatten(start_dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        out = torch.cat([z, z2], dim=0)
        return out

    def update(self, optim, **kwargs):
        x = kwargs['output']
        batch_size = x.size(0)

        # cosine
        scores = torch.mm(x, x.t()).div_(self.tau)

        # euclid
        # x_ref = x.unsqueeze(0).expand(batch_size, -1, -1)
        # c_cmp = x.unsqueeze(1).expand(-1, batch_size, -1)
        # scores = -torch.pow(x_ref - x_cmp, 2).sum(2).div_(self.tau)

        # remove diagonal
        non_diag = torch.LongTensor([[i for i in range(batch_size) if i != j] for j in range(batch_size)]).to(x.device)
        scores = torch.gather(scores, 1, non_diag)

        # select matching (augmented) pairs as targets for classification
        target = torch.arange(batch_size // 2, dtype=torch.long)
        target = torch.cat([target + batch_size // 2 - 1, target], dim=0).to(x.device)
        loss = F.cross_entropy(scores, target)

        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss

    # Comment out the following method to train a linear classifier on the downstream tasks
    @staticmethod
    def train_classifier(z, n_way, n_support, n_query):
        # Assign classes according to cosine similarity to prototypes (normalized mean of support examples per class)
        z = F.normalize(z.view(n_way * (n_support + n_query), -1), p=2, dim=1)
        z = z.view(n_way, n_support + n_query, -1)

        z_support = z[:, :n_support]
        z_query = z[:, n_support:]

        z_proto = z_support.contiguous().view(n_way, n_support, -1).mean(1)
        z_proto = F.normalize(z_proto, p=2, dim=1)
        z_query = z_query.contiguous().view(n_way * n_query, -1)

        scores = torch.mm(z_query, z_proto.t())
        return scores


class ProtoNet(BaseClass):
    """This code is modified from https://github.com/jakesnell/prototypical-networks"""
    def __init__(self, backbone, in_channels, image_size, z_dim, n_support):
        super(ProtoNet, self).__init__()
        self.n_support = n_support
        add_fc = True if image_size >= 32 else False  # add fully connected layer on top for larger images
        last_relu = True
        self.encoder = Encoder(backbone, in_channels, image_size, z_dim, add_fc=add_fc, last_relu=last_relu)

    def encode(self, x, last_layer=None):
        return self.encoder(x, last_layer)

    def forward(self, **kwargs):
        x = kwargs['img']
        n_way = x.size(0)
        n_query = x.size(1) - self.n_support

        x = x.view(n_way * (self.n_support + n_query), *x.size()[2:])
        z = self.encode(x).flatten(start_dim=1)
        z = z.view(n_way, self.n_support + n_query, -1)

        scores = self.train_classifier(z, n_way, self.n_support, n_query)
        target = torch.from_numpy(np.repeat(range(n_way), n_query)).to(x.device)
        out = {'scores': scores, 'target': target}
        return out

    @staticmethod
    def update(optim, **kwargs):
        output = kwargs['output']
        loss = F.cross_entropy(output['scores'], output['target'])

        optim.zero_grad()
        loss.backward()
        optim.step()
        return loss

    # Comment out the following method to train a linear classifier on the downstream tasks
    @staticmethod
    def train_classifier(z, n_way, n_support, n_query):
        # Assign classes according to euclidean distance to prototypes (mean of support examples per class)
        z_support = z[:, :n_support]
        z_query = z[:, n_support:]

        z_support = z_support.contiguous()
        z_proto = z_support.view(n_way, n_support, -1).mean(1)
        z_query = z_query.contiguous().view(n_way * n_query, -1)

        # Compute neg. distance between each prototype (n_classes x z_dim) and each query (n_query x z_dim)
        z_proto = z_proto.unsqueeze(0).expand(n_way * n_query, -1, -1)
        z_query = z_query.unsqueeze(1).expand(-1, n_way, -1)
        scores = -torch.pow(z_proto - z_query, 2).sum(2)
        return scores


if __name__ == '__main__':
    pass
