import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import trajnetplusplustools

from utils import read_file, poly_fit, seq_to_graph, read_ndjson_file


class DatasetTrajnetPP(Dataset):
    """torch.Dataset alternative to the original data, in Trajnet++ format. For more information see:
    https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge"""

    def __init__(self, data_location, obs_len=9, pred_len=12, min_ped=1, norm_lap_matr=True,
                 consider_partial_trajectories=True):
        super(DatasetTrajnetPP, self).__init__()

        self.data_location = data_location
        self.consider_partial_trajectories = consider_partial_trajectories
        self.norm_lap_matr = norm_lap_matr

        self.total_num_seqs = 0  # total number of sequences of trajectories spanned across time.

        if os.path.isdir(self.data_location):
            self.all_files = sorted(os.listdir(self.data_location))
            self.all_files = [os.path.join(self.data_location, _path) for _path in self.all_files]
        else:
            # is a single file
            self.all_files = [self.data_location]

        self.num_peds_in_seq = []
        self.obs_seq_list = []
        self.obs_seq_list_rel = []
        self.pred_seq_list = []
        self.pred_seq_list_rel = []
        self.seq_start_end_list = []

        self.num_distinct_primary_pedestrians = 0
        self.num_discarded_trajectories = 0
        self.seen_primary_ped = {}

        for path in self.all_files:
            data_reader = read_ndjson_file(path)
            self.seen_primary_ped[path] = []
            for scene_id, scene in data_reader.scenes():
                ped_id_list = [t[0].pedestrian for t in scene]
                primary_ped_id = scene[0][0].pedestrian
                if primary_ped_id not in self.seen_primary_ped[path]:
                    self.num_distinct_primary_pedestrians += 1
                    self.seen_primary_ped[path].append(primary_ped_id)
                data = trajnetplusplustools.Reader.paths_to_xy(scene)
                # tensor of shape [obs_len+pred_len,num_peds,2]
                seq = torch.from_numpy(data).to(torch.float)
                if seq.shape[0] != (obs_len + pred_len):
                    raise Exception(f'Got trajectory length {seq.shape[0]}, but expected length of '
                                    f'{obs_len}+{pred_len}={obs_len + pred_len}')
                if not self.consider_partial_trajectories:
                    # discard the trajectories that have nan values
                    seq = seq[:, torch.any(torch.all(~torch.isnan(seq), dim=0), dim=1), :]
                num_peds = seq.shape[1]
                if num_peds <= min_ped:
                    self.num_discarded_trajectories += 1
                    continue  # just one ped, and should have at least one ped + one neighbour
                seq = seq.permute(1, 2, 0)  # (seq_len, num_peds, 2) -> (num_peds, 2, seq_len)
                seq_rel = torch.zeros_like(seq)
                seq_rel[:, :, 1:] = seq[:, :, 1:] - seq[:, :, :-1]
                self.obs_seq_list.append(seq[:, :, :obs_len])
                self.obs_seq_list_rel.append(seq_rel[:, :, :obs_len])
                self.pred_seq_list.append(seq[:, :, obs_len:])
                self.pred_seq_list_rel.append(seq_rel[:, :, obs_len:])
                # primary pedestrian will always be the one with id 0 - the 'start' in seq_start_end
                # in trajnetpp repositories - seq_start_end <=> batch_split
                if self.seq_start_end_list:
                    new_start = self.seq_start_end_list[-1][-1]
                    self.seq_start_end_list.append([new_start, new_start + num_peds])
                else:
                    self.seq_start_end_list.append([0, num_peds])
                self.num_peds_in_seq.append(num_peds)
                self.total_num_seqs += 1

        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.obs_seq_list))
        for ss in range(len(self.obs_seq_list)):
            pbar.update(1)
            v_, a_ = seq_to_graph(self.obs_seq_list[ss], self.obs_seq_list_rel[ss], self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = seq_to_graph(self.pred_seq_list[ss], self.pred_seq_list_rel[ss], self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()
        if self.num_discarded_trajectories > 0:
            print(f"INFO: Discarded {self.num_discarded_trajectories} for not having at least one neighbour")

    def __len__(self):
        return self.total_num_seqs

    def __getitem__(self, index):
        out = [
            self.obs_seq_list[index], self.pred_seq_list[index],
            self.obs_seq_list_rel[index], self.pred_seq_list_rel[index],
            torch.tensor(0), torch.tensor(0),  # non_linear_ped and loss_mask attributes are not going to used
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]
        ]
        return out


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='\t', norm_lap_matr=True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        # Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.v_pred = []
        self.A_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]

        ]
        return out
