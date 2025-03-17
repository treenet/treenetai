import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle as pk

import data_processing_library as dpl


class DatasetConfiguration(object):
    def __init__(self,
                 data_file_path,
                 metadata_file_path,
                 tfrecords_dir_path,
                 segment_length,
                 time_resolution,
                 data_channels,
                 file_id,
                 data_split,
                 random_state,
                 gap_size,
                 gap_type,
                 experiment_type,
                 normalization,
                 channels_to_fix,
                 file_type,
                 tree_number,
                 combination_samples,
                 combination_samples_rand,
                 tree_species
                 ):

        self.metadata_path = metadata_file_path
        self.data_path = data_file_path
        self.tfrecords_path = tfrecords_dir_path
        self.segment_length = segment_length
        self.time_resolution = time_resolution
        self.data_channels = data_channels
        self.file_id = file_id
        self.train_path = self.tfrecords_path+'/train_'+str(file_id)+'.tfrecords'
        self.test_path = self.tfrecords_path+'/validation_'+str(file_id)+'.tfrecords'
        self.data_split = data_split
        self.random_state = random_state
        self.gap_size = gap_size
        self.gap_type = gap_type
        self.experiment_type = experiment_type
        self.normalization = normalization
        self.channels_to_fix = channels_to_fix
        self.file_type = file_type
        self.trees = tree_number
        self.combination_samples = combination_samples
        self.combination_samples_rand = combination_samples_rand
        self.tree_species = tree_species

        print('data channels considered: ', data_channels)
        # NOTE: data_channels is a list of strings
        print('Normalization: ', self.normalization)
        print('data path: ', self.data_path)
        print('experiment type: ', self.experiment_type)
        print('file type: ', self.file_type)

    def segmentation(self):
        metadata = dpl.load_dataframe(self.metadata_path, self.file_type, "metadata")
        df = dpl.load_dataframe(self.data_path, self.file_type, "dfAll")

        # metadata = dpl.clean_metadata(meta)
        # metadata = meta

        if self.experiment_type == 'gap-filling':
            # TODO: the following line removes rare species.
            #  This should be automated and given as an option when running the code. A flag should be added.
            index_names = metadata.loc[
                metadata['species'].isin(['Sorbus aria', 'Larix decidua', 'Quercus robur'])].series_name.to_list()

            metadata = metadata[metadata.series_name.isin(index_names) == False].reset_index(drop=True)

            segments = []
            print('constructing segments of desired length with gaps...')
            # TODO: test this part of the function and confirm that it works.
            for value in df:
                temp = dpl.make_segments(value, self.segment_length, self.time_resolution,
                                         self.normalization, self.trees)
                for el in temp:
                    # Note: At this point the dataframe has been converted to a numpy array
                    segments.extend(
                        [dpl.add_gaps(el, self.segment_length, self.gap_size, self.gap_type, self.channels_to_fix),
                        temp,
                        metadata
                        ]
                    )
                    # Note: The output looks like the following:
                    #  [['LM with gaps', 'weather data with gaps', 'soil data with gaps'],
                    #   ['LM', 'weather data', 'soil data'],
                    #   [metadata]
                    #  ]
                    # NOTE: The format of the list is [[sample1_data_np.array, sample1_label_np.array, sample1_labels_np.array],
                    #                                  [sample2_data_np.array, sample2_label_np.array, sample2_labels_np.array],
                    #                                  [sample3_data_np.array, sample3_label_np.array, sample3_labels_np.array],
                    #                                  ....]

        elif self.experiment_type == 'nearest-neighbours': # TODO: complete this function or remove it
            df, metadata = dpl.nn_signal_preparation(df, metadata)  # TODO: Add the number of neighbours to the script.

            # Todo: the following function should be accessed from dpl and not ts.
            # segments = dpl.make_segments(df, metadata, self.segment_length, self.time_resolution, self.data_channels,
            #                              self.gap_size, self.gap_type, self.experiment_type, self.normalization,
            #                              self.channels_to_fix)

        elif self.experiment_type == 'reconstruction':
            print('constructing dataframe with all possible signal permutations...')
            # NOTE: it is necessary to first add time-series permutations to the original dataframe
            df_with_permutations, new_meta_data = dpl.multi_dendro_channel(df, metadata, self.tree_species, self.trees,
                                                                           self.combination_samples,
                                                                           self.combination_samples_rand)
            # note: the data (df_with_permutations) is a list and new_meta_data is a dictionary

            segments = []
            print('constructing segments of desired length from the permutation dataframe...')
            # TODO: test this part of the function and confirm that it works.
            for value in df_with_permutations:
                temp = dpl.make_segments(value, self.segment_length, self.time_resolution,
                                                  self.normalization, self.trees)
                
                # Note: At this point the dataframe has been converted to a numpy array
                segments.extend([np.delete(temp, self.trees, axis=1),
                    temp[:, self.trees, None],
                    # metadata #TODO: add metadata also
                    ]
                )
                # Note: The output looks like the following:
                #  [['LM_0','LM_1',...,'LM_{trees-1}', 'weather data', 'soil data'],
                #   ['LM_trees'],
                #   ['metadata']
                #  ]
        else:
            raise Exception(f'The experiment type ' + self.experiment_type + ' is unknown.')

        
        # NOTE: Write the processed data to a pickle file
        print('writing pkl file...')
        with open(self.tfrecords_path+'/data_' + str(FLAGS.file_id) + '.pkl', 'wb') as f:
            pk.dump(segments, f)  # Note: this warning will disapear as soon as the if statement is completed.

        return segments

    def write_tfrecords(self):
        print('writing TFrecords...')
        segments = self.segmentation()

        train, validation = train_test_split(segments, test_size=self.data_split, random_state=self.random_state)

        print('writing train and validation data to pkl')
        with open(self.tfrecords_path+'/train_' + str(FLAGS.file_id) + '.pkl', 'wb') as f:
            pk.dump(train, f)
        with open(self.tfrecords_path+'/validation_' + str(FLAGS.file_id) + '.pkl', 'wb') as f:
            pk.dump(validation, f)

        print('writing train segments to TFrecord file...')
        # note: see https://pub.towardsai.net/writing-tfrecord-files-the-right-way-7c3cee3d7b12
        with tf.io.TFRecordWriter(self.train_path) as writer:  # Todo: merge the two for cycles below into one
            for segment in train:
                # NOTE: The segment list contains three items : a multidimensional numpy array (the multichannel
                #  input time series) or segment[0], the multichannel label time series
                #  and a list of metadata values, which are used as the labels for each time series i.e. segment[1].
                features = {
                    'data/timeseries_input': dpl.get_feature(dpl.serialize_array(segment[0])),
                    'label/timeseries_label': dpl.get_feature(dpl.serialize_array(segment[1])),
                    'other/metadata': dpl.get_feature(dpl.serialize_array(segment[2])),
                    # Note: For more details look at
                    #  Ref: https://stackoverflow.com/questions/47861084/how-to-store-numpy-arrays-as-tfrecord
                }
                # labels = segment[2]
                # for col, row in labels.items():
                #    # TODO: This 'for loop' can be part of a 'get_schema()' function in the pandas2tfrecords file
                #    features['label/'+col] = dpl.get_feature(row)

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())

        print('writing validation segments to TFrecord file...')
        with tf.io.TFRecordWriter(self.test_path) as writer:
            for segment in validation:
                features = {
                    'data/timeseries_input': dpl.get_feature(dpl.serialize_array(segment[0])),
                    'label/timeseries_label': dpl.get_feature(dpl.serialize_array(segment[1])),
                    'other/metadata': dpl.get_feature(dpl.serialize_array(segment[2])),
                }

                # labels = segment[2]
                # for col, row in labels.items():
                #    # TODO: This 'for loop' can be part of a 'get_schema()' function in the pandas2tfrecords file
                #    features['label/'+col] = dpl.get_feature(row)

                tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(tf_example.SerializeToString())


def main(_argv):

    configuration = {
        'segment length': FLAGS.segment_length,
        'time resolution': FLAGS.time_resolution,
        'data channels': FLAGS.data_channels,
        'file ID': FLAGS.file_id,
        'split ratio': FLAGS.data_split,
        'random state': FLAGS.random_state,
        'gap size': FLAGS.gap_size,
        'gap type': FLAGS.gap_type,
        'experiment type': FLAGS.experiment_type,
        'normalization': FLAGS.normalization,
        'channels to fix': FLAGS.channels_to_fix,
        'number of dendrometer signals': FLAGS.tree_number,
        'combination_samples': FLAGS.combination_samples,
        'combination_samples_rand': FLAGS.combination_samples_rand,
        'species used': FLAGS.species,
        'note': FLAGS.notes
    }

    try:
        file = open(FLAGS.tfrecords_dir_path+'/info_'+str(FLAGS.file_id)+'.txt', 'wt')
        for a, b in configuration.items():
            file.write(str(a)+': '+str(b)+'\n')
        file.close()
    except NameError:
        print("Unable to write to file")

    print("Configuration: ", configuration)

    data_config = DatasetConfiguration(data_file_path=FLAGS.data_file_path,
                                       metadata_file_path=FLAGS.metadata_file_path,
                                       tfrecords_dir_path=FLAGS.tfrecords_dir_path,
                                       segment_length=FLAGS.segment_length,
                                       time_resolution=FLAGS.time_resolution,
                                       data_channels=FLAGS.data_channels,
                                       file_id=FLAGS.file_id,
                                       data_split=FLAGS.data_split,
                                       random_state=FLAGS.random_state,
                                       gap_size=FLAGS.gap_size,
                                       gap_type=FLAGS.gap_type,
                                       experiment_type=FLAGS.experiment_type,
                                       normalization=FLAGS.normalization,
                                       channels_to_fix=FLAGS.channels_to_fix,
                                       file_type=FLAGS.file_type,
                                       tree_number=FLAGS.tree_number,
                                       combination_samples=FLAGS.combination_samples,
                                       combination_samples_rand=FLAGS.combination_samples_rand,
                                       tree_species=FLAGS.species,
                                      )

    data_config.write_tfrecords()


if __name__ == "__main__":

    from absl import app, flags
    from absl.flags import FLAGS

    flags.DEFINE_string         ('data_file_path', '../data',
                                 'path to directory, images and labels')
    flags.DEFINE_string         ('metadata_file_path', '../metadata',
                                 'metadata path')
    flags.DEFINE_string         ('tfrecords_dir_path', '../tfrecords',
                                 'tfrecords directory')
    flags.DEFINE_integer        ('segment_length', 30,
                                 'length of the segments into which the time series should be divided')
    flags.DEFINE_integer        ('time_resolution', 1,
                                 'time resolution of the time series')
    flags.DEFINE_list           ('data_channels', None,
                                 'select the measured features to be used')
    flags.DEFINE_integer        ('file_id', np.random.randint(1000),
                                 'identifier for each tfrecords file. details about the file contents can be found'
                                 'in the info file with the same ID number.')
    flags.DEFINE_float          ('data_split', 0.2,
                                 'percentage of test data vs train data. The number should be between 0 and 1.')
    flags.DEFINE_integer        ('random_state', 48,
                                 'random state for train-test shuffle')
    flags.DEFINE_integer        ('gap_size', 10,
                                 'average gap size in days')
    flags.DEFINE_string         ('gap_type', 'constant',
                                 'gap size distribution')
    flags.DEFINE_string         ('experiment_type', 'gap-filling',
                                 'type of experiment to be performed. the label also depends on it.'
                                 'choices: gap-filling, time-series-enhancement, nearest-neighbours')
    flags.DEFINE_bool           ('normalization', False,
                                 'should the input time series segments be normalized or not?')
    flags.DEFINE_multi_integer  ('channels_to_fix', 0,
                                 'channels that are considered for gap filling. '
                                 'make sure that the indices correspond to the correct channel.'
                                 '0 should always represent the dendrometer signal.')
    flags.DEFINE_string         ('file_type', 'pkl',
                                 'Type of file where data is stored.'
                                 'choices: rda, csv, npy, pkl')
    flags.DEFINE_integer        ('tree_number', 3,
                                 'number of different tree dendrometer signals to use as input')
    flags.DEFINE_list           ('species', 'all', 'species considered')
    flags.DEFINE_integer        ('combination_samples', '3', 'number input signal combinations (sets) to be selected')
    flags.DEFINE_bool           ('combination_samples_rand', False,
                                 'should the selection be random?')
    flags.DEFINE_string         ('notes', None,
                                 'additional notes for clarification')

    app.run(main)
