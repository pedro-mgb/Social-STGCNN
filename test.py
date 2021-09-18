import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist

from data import TrajectoryDataset, DatasetTrajnetPP
from metrics import *
from model import social_stgcnn
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--num_samples', default=20, type=int, help='Number of samples to generate, per trajectory')
parser.add_argument('--trajnetpp', action='store_true', help='Use data in Trajnet++ format')
parser.add_argument('--use_partial_trajectories', action='store_true',
                    help='If specified with --trajnetpp flag, will consider partial trajectories for neighbours '
                         '(to avoid having to deal with NaNs).')
parser.add_argument('--eval_mode', default='min', type=str, choices=['min', 'max'],
                    help='Mode for multimodal evaluation. min will retrieve best ADE/FDE, max will get worse.')
parser.add_argument('--kde_nll', action='store_true',
                    help='Also compute Kernel Density Estimate Negative Log Likelihood (KDE-NLL) metric')
'''
parser.add_argument('--data_location', default=None, type=str,
                    help='Path to the data to evaluation. By default will evaluate the ETH/UCY models. '
                         'Should be mandatory sent when using --trajnetpp flag. ')
'''
parser.add_argument('--model_tag', default=None, type=str,
                    help='Path to the model evaluation. By default will evaluate the ETH/UCY models. Should be '
                         'mandatory sent when using --trajnetpp flag.')
parser.add_argument('--no_cuda', action='store_true', help='Do not use GPU / CUDA. Instead, do operations on CPU')


def test(KSTEPS=20):
    global loader_test, model, trajnetpp, use_cuda, use_kde_nll, eval_mode_fn
    model.eval()
    ade_bigls = []
    fde_bigls = []
    kde_nll_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        if use_cuda:
            batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0, 2, 3, 1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # if use_cuda:
        #   V_pred= torch.rand_like(V_tr).cuda()

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
        # print(V_pred.shape)

        # For now I have my bi-variate parameters
        # normx =  V_pred[:,:,0:1]
        # normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        if use_cuda:
            cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)

        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 

        # Now sample --num_samples samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        if trajnetpp:
            num_of_objs = 1  # just primary pedestrian

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        k_preds = []

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            # V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        if use_kde_nll:
            if trajnetpp:
                all_step_preds = np.concatenate([arr[:, np.newaxis, 0:1] for arr in raw_data_dict[step]['pred']],
                                                axis=1)
                nll = kde_nll(all_step_preds, raw_data_dict[step]['trgt'][:, 0:1])
            else:
                all_step_preds = np.concatenate([arr[:, np.newaxis] for arr in raw_data_dict[step]['pred']],
                                                axis=1)
                nll = kde_nll(all_step_preds, raw_data_dict[step]['trgt'])
        else:
            nll = None

        for n in range(num_of_objs):
            # get best (eval_mode_fn == min) or worst (eval_mode = max) ADE. Get FDE for the same trajectory
            # This is slightly different from original S-STGCNN code, that picked the best of both, even if they
            # belonged to the same trajectory.
            ade_idx = eval_mode_fn(range(len(ade_ls[n])), key=lambda k: ade_ls[n][k])
            ade_bigls.append(ade_ls[n][ade_idx])
            fde_bigls.append(fde_ls[n][ade_idx])
            if use_kde_nll:
                kde_nll_bigls.append(nll[n])

    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    nll_ = sum(kde_nll_bigls) / len(kde_nll_bigls) if use_kde_nll else None
    return ade_, fde_, nll_, raw_data_dict


args = parser.parse_args()
trajnetpp = args.trajnetpp
use_cuda = not args.no_cuda
use_partial_trajectories = args.use_partial_trajectories
use_kde_nll = args.kde_nll
eval_mode_fn = min if args.eval_mode == 'min' else max

if args.trajnetpp:
    if args.model_tag is None:
        raise Exception('For --trajnetpp, --model_tag must be supplied.')
    paths = ['checkpoint/' + args.model_tag]
else:
    paths = ['./checkpoint/*social-stgcnn*']
KSTEPS = args.num_samples

print("*" * 50)
print('Number of samples:', KSTEPS)
print("*" * 50)

for feta in range(len(paths)):
    ade_ls = []
    fde_ls = []
    nll_ls = []
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:', exps)

    for exp_path in exps:
        print("*" * 50)
        print("Evaluating model:", exp_path)

        model_path = exp_path + '/val_best.pth'
        args_path = exp_path + '/args.pkl'
        with open(args_path, 'rb') as f:
            args = pickle.load(f)

        stats = exp_path + '/constant_metrics.pkl'
        with open(stats, 'rb') as f:
            cm = pickle.load(f)
        print("Stats:", cm)

        # Data prep
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        if trajnetpp and 'trajnetpp' in args.dataset:
            if '21' in args.dataset:
                data_set = 'datasets_in_trajnetpp21/'
            elif '11' in args.dataset:
                data_set = 'datasets_in_trajnetpp11/'
            else:
                raise Exception(f'Dataset in {args.dataset} was not found')
            dset_test = DatasetTrajnetPP(data_set + 'test/', obs_len=obs_seq_len, pred_len=pred_seq_len,
                                         norm_lap_matr=True,
                                         consider_partial_trajectories=use_partial_trajectories)
        else:
            data_set = './datasets/' + args.dataset + '/'

            dset_test = TrajectoryDataset(
                data_set + 'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1, norm_lap_matr=True)

        loader_test = DataLoader(
            dset_test,
            batch_size=1,  # This is irrelative to the args batch size parameter
            shuffle=False,
            num_workers=0)

        # Defining the model
        model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                              output_feat=args.output_size, seq_len=args.obs_seq_len,
                              kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)
        if use_cuda:
            model = model.cuda()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda:0' if use_cuda else 'cpu')))

        ade_ = 999999
        fde_ = 999999
        print("Testing ....")
        ad, fd, nll, raw_data_dic_ = test(KSTEPS)
        ade_ = min(ade_, ad)
        fde_ = min(fde_, fd)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        nll_ls.append(nll)
        if use_kde_nll:
            print("ADE:", ade_, " FDE:", fde_, " KDE-NLL:", nll)
        else:
            print("ADE:", ade_, " FDE:", fde_)

    if len(exps) > 1:  # perform average across all files
        print("*" * 50)
        print("Avg ADE:", sum(ade_ls) / len(exps))
        print("Avg FDE:", sum(fde_ls) / len(exps))
        if use_kde_nll:
            print("Avg KDE-NLL:", sum(nll_ls) / len(exps))
