import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class OcclusionHead(nn.Module):
    # two output layers - according to contflow
    def __init__(self, input_dim=128, hidden_dim=256, architecture=None):
        super(OcclusionHead, self).__init__()
        self.architecture = architecture

        if architecture is None or architecture == 'simple':
            self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
            self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)
        elif architecture == 'morelayers':
            self.model = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 2, 3, padding=1),
            )
        else:
            raise NotImplementedError('This type of architecture is not implemented')

    def forward(self, x):
        if self.architecture is None or self.architecture == 'simple':
            return self.conv2(self.relu(self.conv1(x)))
        else:
            return self.model(x)



class UncertaintyHead(nn.Module):
    # single output layer
    def __init__(self, input_dim=128, hidden_dim=256, architecture=None):
        super(UncertaintyHead, self).__init__()
        self.architecture = architecture

        if architecture is None or architecture == 'simple':
            self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
            self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
            self.relu = nn.ReLU(inplace=True)
        elif architecture == 'morelayers':
            self.model = nn.Sequential(
                nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, 1, 3, padding=1),
            )
        else:
            raise NotImplementedError('This type of architecture is not implemented')

    def forward(self, x):
        if self.architecture is None or self.architecture == 'simple':
            return self.conv2(self.relu(self.conv1(x)))
        else:
            return self.model(x)



class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow, motion_features

class OcclusionAndUncertaintyBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(OcclusionAndUncertaintyBlock, self).__init__()
        self.args = args

        architecture = 'simple'
        if 'morelayers' in args.occlusion_module:
            architecture = 'morelayers'
        self.occlusion_detach = getattr(args, 'occlusion_input_detach', False)
        self.uncertainty_detach = getattr(args, 'uncertainty_input_detach', False)

        self.occl_head = OcclusionHead(hidden_dim, hidden_dim=128, architecture=architecture)

        if 'with_uncertainty' in args.occlusion_module:
            if 'separate' in args.occlusion_module:
                self.uncertainty_head = UncertaintyHead(hidden_dim, hidden_dim=128, architecture=architecture)
            else:
                raise NotImplementedError(f'Type {args.occlusion_module} of occlusion/uncertainty module is not implemented')

    def forward(self, net, inp, corr, flow, delta_flow, motion_features):
        inp = torch.cat([net, inp, corr, flow, delta_flow, motion_features], dim=1)
        orig_inp = inp

        if ('with_uncertainty' not in self.args.occlusion_module) or ('separate' not in self.args.occlusion_module):
            raise NotImplementedError(f'Type {self.args.occlusion_module} of occlusion/uncertainty module is not implemented')

        if self.occlusion_detach:
            inp = orig_inp.detach()
        else:
            inp = orig_inp
        occl = self.occl_head(inp)

        if self.uncertainty_detach:
            inp = orig_inp.detach()
        else:
            inp = orig_inp
        uncertainty = self.uncertainty_head(inp)
        return occl, uncertainty

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow, motion_features



