from models.resnet import ResNet
from models.ca import *
from models.sa import *
from models.resnet18 import resnet18
from models.wrn import WRN28

class RENet(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        # self.encoder = ResNet(args=args)
        # self.encoder = resnet18(args=args)
        self.encoder = WRN28(args=args)
        self.encoder_dim = 640
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)
        self.scr_module = self._make_scr_layer()
        self.match_net = match_block(640)
        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(self.encoder_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def _make_scr_layer(self):
        kernel_size, padding = (5, 5), 2
        layers = list()

        if self.args.self_method == 'sa':
            corr_block = mySelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
        else:
            raise NotImplementedError

        if self.args.self_method == 'sa':
            layers.append(corr_block)
        return nn.Sequential(*layers)

    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            return self.encode(input, False)
        elif self.mode == 'ca':
            spt, qry = input
            return self.ca(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)

    def ca(self, spt, qry):  # 支持，查询

        spt = spt.squeeze(0)  # 移除数组中维度为1的维度

        spt = self.normalize_feature(spt)  # 1
        qry = self.normalize_feature(qry)
        # _________________________________________________________________________________baseline
        # way = spt.shape[0]
        # num_qry = qry.shape[0]
        # H_s, W_s, H_q, W_q = 5,5,5,5
        # spt_c = F.normalize(spt, p=2, dim=1, eps=1e-8)
        # qry_c = F.normalize(qry, p=2, dim=1, eps=1e-8)
        # # way , C , H_s , W_s --> num_qry * way, C , H_s , W_s
        # # num_qry , C , H_q , W_q --> num_qry * way,C ,H_q , W_q
        # spt_c = spt_c.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)
        # qry_c = qry_c.unsqueeze(1).repeat(1, way, 1, 1, 1)
        # d_s = spt_c.view(num_qry, way,640,H_s*W_s)  # 10，5，25，5，5
        # d_q = qry_c.view(num_qry, way,640,H_q*W_q)  # 10，5，5，5，25
        # d_s = self.gaussian_normalize(d_s, dim=3)
        # d_q = self.gaussian_normalize(d_q, dim=3)
        #
        # # applying softmax for each side
        # d_s = F.softmax(d_s / self.args.temperature_attn, dim=3)
        # d_s = d_s.view(num_qry, way,640,H_s, W_s)  # 10，5，5，5，5，5
        # d_q = F.softmax(d_q / self.args.temperature_attn, dim=3)
        # d_q = d_q.view(num_qry, way,640,H_q, W_q)  # 10，5，5，5，5，5
        #
        # spt_attended = d_s * spt.unsqueeze(0)  # 10，5，640，5，5
        # qry_attended = d_q * qry.unsqueeze(1)  # 10，5，640，5，5

        # _____________________________________________________________________________________
        way = spt.shape[0]
        num_qry = qry.shape[0]
        H_s, W_s, H_q, W_q = 6,6,6,6
        spt_attended1, qry_attended1 = self.match_net(spt, qry)  # 先 Channel
        spt_attended1 = spt_attended1.view(num_qry, way, 640, H_s, W_s)
        qry_attended1 = qry_attended1.view(num_qry, way, 640, H_q, W_q)

        spt_attended1 = F.relu(spt_attended1, inplace=True)
        qry_attended1 = F.relu(qry_attended1, inplace=True)

        d_s = spt_attended1.view(num_qry, way, 640, H_s * W_s)  # 10，5，25，5，5
        d_q = qry_attended1.view(num_qry, way, 640, H_q * W_q)  # 10，5，5，5，25

        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        d_s = self.gaussian_normalize(d_s, dim=3)
        d_q = self.gaussian_normalize(d_q, dim=3)

        # applying softmax for each side
        d_s = F.softmax(d_s / self.args.temperature_attn, dim=3)
        d_s = d_s.view(num_qry, way, 640, H_s, W_s)  # 10，5，5，5，5，5
        d_q = F.softmax(d_q / self.args.temperature_attn, dim=3)
        d_q = d_q.view(num_qry, way, 640, H_q, W_q)  # 10，5，5，5，5，5

        spt_attended = d_s * spt.unsqueeze(0)  # 10，5，640，5，5
        qry_attended = d_q * qry.unsqueeze(1)  # 10，5，640，5，5

        # averaging embeddings for k > 1 shots
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        # In the main paper, we present averaging in Eq.(4) and summation in Eq.(5).
        # In the implementation, the order is reversed, however, those two ways become eventually the same anyway :)
        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])
        qry_pooled = qry.mean(dim=[-1, -2])
        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_pooled)
        else:
            return similarity_matrix / self.args.temperature

# ----------------------------------------------------------------------------------
    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)  # 求dim上的方差
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))  # （x原始-x平均）/根号下x_var
        return x

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)  # x-x.mean(1)行求平均值并在channal维上增加一个维度

    def encode(self, x, do_gap=True):
        x = self.encoder(x)

        if self.args.self_method:
            identity = x  # (80,640,5,5)
            x = self.scr_module(x)
            if self.args.self_method == 'sa':
                x = x + identity
            x = F.relu(x, inplace=True)

        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x


