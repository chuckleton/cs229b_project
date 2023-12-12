from pathlib import Path
import torch
import matplotlib.pyplot as plt
import scipy.signal

import sys
sys.path.insert(0, '/root/test_data_management/cs229b/state-spaces')

# proprietary information in this file, not included in the repo
import src.dataloaders.cs229b.load_tests as load_tests

from src.dataloaders.base import SequenceDataset

import numpy as np

import tqdm
import zlib

plt.switch_backend('agg')

class FullTestBaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 test_filters: load_tests.GetTestQueryFilters,
                 data_channels: list[load_tests.DatasetChannel],
                 control_channels: list[load_tests.DatasetChannel],
                 interpolate_to_channel: load_tests.DatasetChannel,
                 sample_time: float = 2.0,
                 pre_sample_time: float = 0.25,
                 device: str = "cpu",
                 length: int = 5000,
                ):
        self.length = length
        self.test_filters = test_filters
        self.test_ids = load_tests.get_test_ids(self.test_filters)
        self.saved_tests = [load_tests.get_test(test_id) for test_id in self.test_ids]
        self.test_weights = self.get_test_weights()
        print(f"{len(self.test_ids)=}")
        total_test_on_time = sum([test.startup_to_shutdown_seconds for test in self.saved_tests])
        print(f"{total_test_on_time=}")
        test_identifiers = [test.test_identifier for test in self.saved_tests]
        print(f"{test_identifiers=}")
        self.data_channels = data_channels
        self.data_channel_names = [channel.name for channel in self.data_channels]
        self.control_channels = control_channels
        self.control_channel_names = [channel.name for channel in self.control_channels]
        self.all_channels = self.data_channels + self.control_channels
        self.interpolate_to_channel = interpolate_to_channel
        self.sample_time = sample_time
        self.pre_sample_time = pre_sample_time
        self.device = device

        self.init_dataset()
        self.set_correct_dtypes()

    def get_test_weights(self):
        # Weight each test slightly based on the startup_to_shutdown_seconds
        # so that we don't sample too heavily from the shorter tests

        # Get the startup_to_shutdown_seconds for each test
        startup_to_shutdown_seconds = [test.startup_to_shutdown_seconds for test in self.saved_tests]
        # Get the mean and standard deviation of the startup_to_shutdown_seconds
        mean = np.mean(startup_to_shutdown_seconds)
        std = np.std(startup_to_shutdown_seconds)
        if std == 0:
            # If the std is 0, then all the tests have the same length,
            # so we don't need to weight them
            return np.ones(len(self.saved_tests)) / len(self.saved_tests)
        # Calculate the weights
        # using a gaussian distribution with mean and std
        weights = np.exp(-(startup_to_shutdown_seconds - mean)**2 / (2 * std**2))
        # Normalize the weights
        weights = weights / np.sum(weights)
        # print(f"{weights=}")
        return weights

    def set_correct_dtypes(self):
        # we want torch.float32
        self.control_data = self.control_data.float()
        self.data_data = self.data_data.float()

    def init_dataset(self):
        test_control, test_data = self.get_initial_item(0)
        self.control_data = np.empty(
            (len(self), test_control.shape[0], test_control.shape[1]),
        )
        self.data_data = np.empty(
            (len(self), test_data.shape[0], test_data.shape[1]),
        )
        print(f"Initializing dataset of length {len(self)}")

        # Check if the dataset has already been saved to a file
        saved_data_path = Path("/root/test_data_management/cs229b/saved_data")
        identifier = f"{len(self)=}_{self.sample_time=}_{self.pre_sample_time=}_{self.data_channel_names=}_{self.control_channel_names=}_{self.test_ids=}"
        control_data_fname = f"control-full_test_dataset_{zlib.adler32(identifier.encode('utf-8'))}.npz"
        data_fname = f"data-full_test_dataset_{zlib.adler32(identifier.encode('utf-8'))}.npz"
        control_data_path = saved_data_path / control_data_fname
        data_path = saved_data_path / data_fname
        try:
            self.control_data = np.load(control_data_path, allow_pickle=True)["data"]
            # Convert the data to a tensor
            self.control_data = torch.from_numpy(self.control_data).to(self.device)
            print(f"Loaded controls dataset of length {len(self)} from file {control_data_path}")
            self.data_data = np.load(data_path, allow_pickle=True)["data"]
            # Convert the data to a tensor
            self.data_data = torch.from_numpy(self.data_data).to(self.device)
            print(f"Loaded data dataset of length {len(self)} from file {data_path}")
            return
        except FileNotFoundError:
            print("No dataset files found, creating new dataset")

        pbar = tqdm.tqdm(range(len(self)))
        for i in pbar:
            # get a random test index, weighted by self.test_weights
            test_idx = np.random.choice(len(self.test_ids), p=self.test_weights)
            # Set the tqdm description
            pbar.set_description(f"Loading test {test_idx}")
            # Get the test data
            control_data, data = self.get_initial_item(test_idx)
            # print(f"{control_data.shape=}", f"{data.shape=}")
            # Save the data
            self.control_data[i] = control_data.cpu().numpy()
            self.data_data[i] = data.cpu().numpy()

        # Save the datasets to a file
        np.savez_compressed(control_data_path, data=self.control_data)
        print(f"Saved controls dataset of length {len(self)} to file {control_data_path}")
        np.savez_compressed(data_path, data=self.data_data)
        print(f"Saved data dataset of length {len(self)} to file {data_path}")

        # Convert the data to a tensor
        self.control_data = torch.from_numpy(self.control_data).to(self.device)
        self.data_data = torch.from_numpy(self.data_data).to(self.device)

    def __len__(self):
        return self.length

    def get_initial_item(self, idx: int):
        # Get the test.
        test = self.saved_tests[idx]
        # print(f"{test.test_identifier=}")
        # Get the data for the test
        test_data = load_tests.get_test_data_numpy(
            test, self.all_channels, self.sample_time, self.pre_sample_time)
        # Convert the data to a dict of tuples of tensors (time_data, data).
        test_data = {
            channel: (
                torch.from_numpy(test_data[channel][0]).to(self.device),
                torch.from_numpy(test_data[channel][1]).to(self.device)
            ) for channel in self.all_channels}
        # Interpolate the data to the specified channel using torch.nn.functional.interpolate
        interpolate_to_length = test_data[self.interpolate_to_channel][0].shape[0]
        for channel in self.all_channels:
            if channel != self.interpolate_to_channel:
                # If the channel.transformation is PERIODOGRAM, we take a periodogram over each window at the rate
                # of the interpolate_to_channel
                if channel.transformation == load_tests.DatasetChannelTransformation.PERIODOGRAM:
                    channel_fs = 1 / np.mean(np.diff(test_data[channel][1]))
                    test_data[channel] = test_data[channel][1]
                    # Get the number of sample per window, as well as the number of windows
                    nperseg = len(test_data[channel]) // interpolate_to_length
                    # print(f"{nperseg=}")
                    # print(f"{len(test_data[channel])=}")
                    # print(f"{interpolate_to_length=}")
                    periodograms = []
                    # Take the periodogram
                    for seg_num in range(interpolate_to_length):
                        periodogram = scipy.signal.periodogram(
                            test_data[channel][seg_num*nperseg:(seg_num+1)*nperseg],
                            fs=channel_fs,
                        )
                        periodograms.append(periodogram[1])
                    # print(f"{len(periodograms)=}")
                    # print(f"{len(periodograms[0])=}")
                    # Transform the periodograms into a tensor
                    periodograms = torch.tensor(periodograms).to(self.device)
                    # print(f"{periodograms.shape=}")
                    test_data[channel] = periodograms
                else:
                    # only interpolate the data, not the time
                    # reshape to (1, 1, -1) to match the shape expected by interpolate
                    test_data[channel] = test_data[channel][1].reshape(1, 1, -1)
                    test_data[channel] = torch.nn.functional.interpolate(test_data[channel], size=interpolate_to_length, mode='linear')
                    # reshape back to (1, -1) for the data
                    test_data[channel] = test_data[channel].reshape(-1, 1)
            else:
                test_data[channel] = test_data[channel][1].reshape(-1, 1)
        # Get the control data for the test
        test_control_data = {channel: test_data[channel] for channel in self.control_channels}
        # Merge the control data into a single tensor
        test_control_data = torch.cat([test_control_data[channel] for channel in self.control_channels], dim=1)
        # Get the data for the test
        test_data = {channel: test_data[channel] for channel in self.data_channels}
        # Merge the data into a single tensor
        test_data = torch.cat([test_data[channel] for channel in self.data_channels], dim=1)
        # Return the data
        # print(f'{test_control_data.shape=}', f'{test_data.shape=}')
        return test_control_data, test_data

    def __getitem__(self, idx: int):
        # print(f"{self.control_data[idx]=}, {self.data_data[idx]=}")
        return self.control_data[idx], self.data_data[idx]

class TestDataset(SequenceDataset):
    _name_="full_test"
    d_output = 1
    l_output = int(2000 * (5) / 2 - 1)

    @property
    def init_defaults(self):
        self.sample_time = 4.0 #seconds
        self.pre_sample_time = 1.0 #seconds

        test_numbers_train = list(range(200))

        # Remove number causing problems
        test_numbers_train.remove(123)
        test_numbers_train.remove(149)

        test_numbers_val = [96, 173]
        test_numbers_test = list(range(2,200))

        # Remove test_numbers_val and test_numbers_test from test_numbers_train
        for test_number in test_numbers_val:
            if test_number in test_numbers_train:
                test_numbers_train.remove(test_number)
        # for test_number in test_numbers_test:
        #     test_numbers_train.remove(test_number)

        self.train_filters = load_tests.GetTestQueryFilters(
            test_filters=load_tests.TestFilters(
                test_numbers=test_numbers_train,
                min_startup_to_shutdown_seconds=self.sample_time + self.pre_sample_time,
            ),
            program_filters=load_tests.ProgramFilters(
                program_names=["A110"],
            ),
            phase_filters=load_tests.PhaseFilters(
                phase_names=["development","qualification"],
            ),
        )

        self.val_filters = load_tests.GetTestQueryFilters(
            test_filters=load_tests.TestFilters(
                test_numbers=test_numbers_val,
                min_startup_to_shutdown_seconds=self.sample_time + self.pre_sample_time,
            ),
            program_filters=load_tests.ProgramFilters(
                program_names=["A110"],
            ),
            phase_filters=load_tests.PhaseFilters(
                phase_names=["development","qualification"],
            ),
        )

        self.test_filters = load_tests.GetTestQueryFilters(
            test_filters=load_tests.TestFilters(
                test_numbers=test_numbers_test,
                min_startup_to_shutdown_seconds=self.sample_time + self.pre_sample_time,
            ),
            program_filters=load_tests.ProgramFilters(
                program_names=["A110"],
            ),
            phase_filters=load_tests.PhaseFilters(
                phase_names=["atp"],
            ),
        )

        fta_channel = load_tests.DatasetChannel(
            name="FTA",
            normalization="SOMETHING",
        )
        ati102_channel = load_tests.DatasetChannel(
            name="ATI102",
            normalization="SOMETHING",
            transformation=load_tests.DatasetChannelTransformation.BINARY,
        )

        self.data_channels = [fta_channel]
        self.control_channels =[
            ati102_channel,
            load_tests.DatasetChannel(
                name="PT222",
                normalization="SOMETHING",
            ),
            load_tests.DatasetChannel(
                name="PT333",
                normalization="SOMETHING",
            ),
        ]
        self.interpolate_to_channel = ati102_channel

        return {}

    @property
    def d_input(self):
        return len(self.control_channels)

    def setup(self):
        self.dataset_train = FullTestBaseDataset(
            self.train_filters,
            self.data_channels,
            self.control_channels,
            self.interpolate_to_channel,
            self.sample_time,
            self.pre_sample_time,
            length=1500,
        )
        self.dataset_val = FullTestBaseDataset(
            self.val_filters,
            self.data_channels,
            self.control_channels,
            self.interpolate_to_channel,
            self.sample_time,
            self.pre_sample_time,
            length=200,
        )
        self.dataset_test = FullTestBaseDataset(
            self.test_filters,
            self.data_channels,
            self.control_channels,
            self.interpolate_to_channel,
            self.sample_time,
            self.pre_sample_time,
            length=200,
        )

if __name__ == "__main__":
    pass