import sys
import os.path as path
import os

utils_path = path.join(path.dirname(path.dirname(os.getcwd())), 'General')
sys.path.append(utils_path)

import seaborn as sns
import numpy as np
import pandas as pd
import pickle

sns.set(style="whitegrid")


def calculate_std(data, column_name, k=10):
    '''
    Calculates the standard deviation of a given columns
    :param data: DataFrame --> data
    :param column_name: String --> Column name for the standard deviation is to be calculated
    :param k: Int --> Window size
    :return: DataFrame --> Data with additional column for the standard deviation (column_name_std)
    '''
    N = data.shape[0]
    var = [0] * N
    data[column_name + '_std'] = var

    n = k

    while n < N:
        if data['file label'].iloc[n] == data['file label'].iloc[n-1]:
            data[column_name + '_std'].iloc[n] = np.std(data[column_name].iloc[n-k:n])
            n += 1
        else:
            n += k

    return data


def center_data(data, pollution_sources, bins, exclude, mean=None):
    '''
    Centers the bin data for each pollution source
    :param data: DataFrame --> Data
    :param pollution_sources: --> List of pollution sources
    :param bins: List of column names for the bins
    :param exclude: List of pollution sources to exclude
    :param mean: If passed, this mean will be used to center the data, for example pass the mean from the training
    set to center the test data
    :return: Centered data and the mean used to center the data
    '''

    data_centered = data[bins].copy()
    data_centered = data_centered.dropna()

    if mean is None:
        mean = np.mean(data_centered)

    data_centered = data_centered - mean
    data_centered['pollution source'] = data['pollution source']

    for i, item in enumerate(pollution_sources):
        if i == 0:
            data_centered_final = data_centered[data['pollution source'] == item]
        else:
            data_centered_final = pd.concat([data_centered_final, data_centered[data['pollution source'] == item]])

    for i in range(len(exclude)):
        data_centered_final = data_centered_final[data_centered_final['pollution source'] != exclude[i]]

    data[bins] = data_centered_final[bins]

    return data, mean


def clean_data(data):
    '''
    Cleans the dataset by removing measurements with data missing
    :param data: DataFrame to be cleaned
    :return: Cleaned dataframe
    '''
    data = data.drop(columns=['luxLevel', 'motion', 'gpsAltitude'])
    data = data.dropna()
    data = data[data['total'] != 0]
    data = data[data['temperature'] != 0]
    data = data[data['humidity'] != 0]
    data = data[data['gpsAccuracy'] != 'None']
    data = data[data['gps_dist'] != 0]
    data = data[data['gps_dist_std'] != 0]

    return data


def distance_euclidean(data):
    '''
    Calculate the Euclidean distance between two GPS points based on the longitude and latitude
    :param data: DataFrame --> Needs to include gpsLongitude and gpsLatitude features
    :return: DataFrame with gps_distance included as additional feature
    '''

    N = data.shape[0]
    gps_dist = [0] * N
    data['gps_dist'] = gps_dist

    for n in range(1, N):
        if data['file label'].iloc[n] == data['file label'].iloc[n-1]:
            data['gps_dist'].iloc[n] = np.sqrt((data['gpsLongitude'].iloc[n-1] - data['gpsLongitude'].iloc[n])**2 \
                                               + (data['gpsLatitude'].iloc[n-1] - data['gpsLatitude'].iloc[n])**2)

    return data


def load_data(path_data, path_save, source_files, bins, pollution_sources):
    '''
    Loads a list of files
    :param path_data: String --> Path to file directory
    :param path_save: String --> Path to save processed files
    :param source_files: List --> List of file names
    :param bins: List --> List of bin values
    :param pollution_sources: List --> List of pollution sources
    :return: Return all of the data
    '''

    file_count = 0

    exclude = []

    with open(path.join(path_data, 'exclude.txt')) as files_exclude:
        for file in files_exclude:
            exclude.append(file.strip('\n'))

    file_label_idx = []
    file_label_name = []
    for source_file in source_files:
        with open(path.join(path_data, source_file), 'r') as files:
            for i, file in enumerate(files):
                file = file.strip('\n')
                print(file)
                file = file.split(',')
                print(file)
                mode = file[-2].strip(' ')
                pollution_source = file[-1].strip(' ')
                if pollution_source in pollution_sources:
                    pollution_source_idx = pollution_source2idx(pollution_source, pollution_sources)
                else:
                    pollution_source_idx = -1
                file = file[0]
                if file not in exclude:
                    if file_count == 0:
                        df = pd.read_csv(path.join(path_data, file+'.csv'))
                    else:
                        df = pd.read_csv(path.join(path_data, file+'.csv'))
                    label = [i] * df.shape[0]
                    label = pd.DataFrame(label, columns=['file label'])
                    file_label_idx.append(i)
                    file_label_name.append(file)
                    pollution_source = [pollution_source] * df.shape[0]
                    pollution_source = pd.DataFrame(pollution_source, columns=['pollution source'])
                    pollution_source_idx = [pollution_source_idx] * df.shape[0]
                    pollution_source_idx = pd.DataFrame(pollution_source_idx, columns=['pollution source idx'])
                    df = pd.concat([df, label, pollution_source, pollution_source_idx], axis=1)
                    if mode == 'o':
                        label = [1] * df.shape[0]
                        io = pd.DataFrame(label, columns=['i/o'])
                        df = pd.concat([df, io], axis=1)
                    elif mode == 'i':
                        label = [0] * df.shape[0]
                        io = pd.DataFrame(label, columns=['i/o'])
                        df = pd.concat([df, io], axis=1)

                    columns = df.columns.tolist()
                    if 'PM1' in columns:
                        columns[columns.index('PM1')] = 'pm1'
                    if 'PM2.5' in columns:
                        columns[columns.index('PM2.5')] = 'pm2_5'
                    if 'PM10' in columns:
                        columns[columns.index('PM10')] = 'pm10'
                    if 'latitude' in columns:
                        columns[columns.index('latitude')] = 'gpsLatitude'
                    if 'longitude' in columns:
                        columns[columns.index('longitude')] = 'gpsLongitude'
                    if 'phoneTimestamp' in columns:
                        columns[columns.index('phoneTimestamp')] = 'timestamp'
                    if 'Timestamp' in columns:
                        columns[columns.index('Timestamp')] = 'timestamp'

                    df.columns = columns

                    if file_count == 0:
                        data = df.copy()
                    else:
                        data = pd.concat([data, df])
                    file_count += 1
                    print(f'loaded {file}, file no. {file_count}')

    data = data.reset_index(drop=True)
    data['total'] = np.sum(data[bins], axis=1)
    data = data[data['total'] != 0]
    if 'environment_index' in data.columns:
        del data['environment_index']
    if 'battery' in data.columns:
        del data['battery']
    if 'validationLabel' in data.columns:
        del data['validationLabel']
    if 'partial_io_label' in data.columns:
        del data['partial_io_label']
    if 'Unnamed: 0' in data.columns:
        del data['Unnamed: 0']
    if 'day_night_labels' in data.columns:
        del data['day_night_labels']
    if 'partial_ps_label' in data.columns:
        del data['partial_ps_label']
    if 'psLabels' in data.columns:
        del data['psLabels']

    data_norm = data.copy()
    data_norm[bins] = data_norm[bins].dropna()
    data_norm = utils.normalise_PSD(data_norm)

    file_meta = list(zip(file_label_idx, file_label_name))
    file_meta = pd.DataFrame(file_meta, columns=['file idx', 'file name'])
    file_meta.to_csv(path.join(path_save, 'file_meta.csv'))

    return data_norm


class StandardScalerDataFrame():
    '''
    Object used to scale a set of features to unit variance and zero means
    '''

    def __init__(self):

        self.labels_scale = None
        self.labels_no_scale = None

    def fit(self, data, labels_scale, labels_no_scale):
        '''
        Fit the standard scalar object
        :param data: Data to scale from
        :param labels_scale: List of DataFrame column names to be scaled
        :param labels_no_scale: List of DataFrame column names not to be scaled
        :return: None
        '''

        means = {}
        std = {}
        for label in labels_scale:
            means[label] = np.mean(data[label], axis=0)
            std[label] = np.std(data[label], axis=0)

        self.means = means
        self.std = std
        self.labels_scale = labels_scale
        self.labels_no_scale = labels_no_scale

    def transform(self, data):
        '''
        Scale a data set based on the fit
        :param data: dataframe to be scaled, only labels which were scaled in fit will be fit
        :return: Dataframe with lables scaled
        '''

        data_scale = data[self.labels_scale + self.labels_no_scale].copy()

        for label in self.labels_scale:
            data_scale[label] = (data[label] - self.means[label]) / self.std[label]

        return data_scale


def split_data(data, split=0.7, shuffle=True, seed=1000):
    '''
    Randomly shuffle then split data into two
    :param data: DataFrame to be split
    :param split: split can either be a decimal fraction or an integer specifiying the number of samples.
    :param shuffle: If true data is shuffled before splitting
    :param seed: Seed for shuffle
    :return: Two dataframes according to split
    '''

    # print(data['file label'])
    np.random.seed(seed)

    N = data.shape[0]

    if shuffle is True:
        idx = np.arange(0, N)
        np.random.shuffle(idx)

        if split <= 1:
            split1_idx = idx[:int(np.floor(split*N))]
            split2_idx = idx[int(np.floor(split*N)):]
        else:
            split1_idx = idx[:split]
            split2_idx = idx[split:]

        data_1 = data.iloc[split1_idx, :]
        data_2 = data.iloc[split2_idx, :]
    elif shuffle is False:
        if split <= 1:
            data_1 = data.iloc[:int(np.floor(split*N)), :]
            data_2 = data.iloc[int(np.floor(split*N)):, :]
        else:
            data_1 = data.iloc[:split, :]
            data_2 = data.iloc[split:, :]


    data_1 = data_1.reset_index(drop=True)
    data_2 = data_2.reset_index(drop=True)

    return data_1, data_2


def split_data_label(data, labels, shuffle=True, seed=1000):
    '''
    Split data frame based on file labels
    :param data: DataFrame to be split
    :param labels: Labels to split data by
    :param shuffle: If True then data is shuffled
    :param seed: Seed used for shuffle
    :return: Data with specifed labels
    '''

    # print(data['file label'])
    np.random.seed(seed)

    # print(pd.unique(data['file label']))

    data_filt = pd.DataFrame([], columns=data.columns)
    for label in labels:
        data_filt = pd.concat([data_filt, data[data['file label'] == label]])


    N = data_filt.shape[0]

    if shuffle is True:
        idx = np.arange(0, N)
        np.random.shuffle(idx)
        data_filt = data_filt.iloc[idx, :]

    return data_filt


def shuffle_data(data, seed=1000):
    '''
    Shuffle DataFrame
    :param data: DataFrame to be shuffled
    :param seed: Int --> Random seed
    :return: shuffled DataFrame
    '''

    np.random.seed(seed)

    N = data.shape[0]
    idx = np.arange(0, N)
    np.random.shuffle(idx)

    return data.iloc[idx, :]

def subsample(data, seed):
    '''
    Subsample data to smallest class based on indoor outdoor label
    :param data: DataFrame to subsample
    :param seed: Int --> Random seed
    :return: Subsampled DataFrame
    '''

    n_in = data[data['i/o'] == 0].shape[0]
    n_out = data[data['i/o'] == 1].shape[0]

    if n_in > n_out:
        data_sub = data[data['i/o'] == 0].sample(n_out, random_state=seed)
        data_sub = pd.concat([data_sub, data[data['i/o'] == 1]], axis=0)
    elif n_out >= n_in:
        data_sub = data[data['i/o'] == 1].sample(n_in, random_state=seed)
        data_sub = pd.concat([data_sub, data[data['i/o'] == 0]], axis=0)

    data_sub = data_sub.sort_values(['file label'])

    return data_sub

def pollution_source2idx(pollution_source, pollution_sources):
    '''
    Function to convert a pollution source to an index
    :param pollution_source: Pollution source to index
    :param pollution_sources: List of pollution sources
    :return: Index of pollution source
    '''
    return pollution_sources.index(pollution_source)

def idx2pollution_source(idx, pollution_sources):
    '''
    Function to convert a pollution source index to pollution source names
    :param idx: Int --> Index of pollution source
    :param pollution_sources: List of pollution sources
    :return: The pollution source
    '''
    return pollution_sources[idx]

def save(obj, pathname, filename):
    '''
    Save and object using pickle
    :param obj: Object to be saved
    :param pathname: String --> Path to save object
    :param filename: Sting --> Name of file to save
    :return: None
    '''
    with open(path.join(pathname,filename), 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load(pathname, filename):
    '''
    Load an object using pickle
    :param pathname: String --> Path to load object
    :param filename: Sting --> Name of file to load
    :return: object which has been loaded
    '''
    with open(path.join(pathname,filename), 'rb') as f:
        return pickle.load(f)
