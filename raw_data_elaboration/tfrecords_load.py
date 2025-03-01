from functools import partial
import tensorflow as tf
from apps.raw_data_elaboration.feature_descriptions import treenet_timeseries_short


def read_tfrecord(example, labelling_pattern):
    # TODO: the tfrecord_feature_format structure should be passed from the very beginning.
    tfrecord_feature_format = treenet_timeseries_short
    example = tf.io.parse_single_example(example, tfrecord_feature_format)
    timeseries_input = tf.io.parse_tensor(example["data/timeseries_input"], out_type=tf.float64)
    # NOTE: restore 2D array from byte string
    print('timeseries_input in function: ', timeseries_input)

    # TODO: the labels have to be organized better so that they work. The labels are loaded automatically,
    #  so this is not the problem. The problem is when a non-matching label is passed through the 2_train.sh file.
    #  This is also related to the config file, which has to be addapted somehow. The labels should be printed onto
    #  the info file when the tfrecords are created. The config file should not have static names of labels. Rather, the
    #  names should be taken from the 2_train.sh file.
    # labels = []
    # if labelling_pattern["timeseries_label"]:
    #     timeseries_label = tf.io.parse_tensor(example["label/timeseries_label"], out_type=tf.float64)
    #     labels.append(timeseries_label)

    # label = tf.stack(labels)

    # return timeseries_input, label
    # TODO: END

    timeseries_label = tf.io.parse_tensor(example["label/timeseries_label"], out_type=tf.float64)
    # timeseries_metadata = tf.io.parse_tensor(example["other/metadata"], out_type=tf.string)
    # TODO: the metadata has to be included in some other way, or there should be a flag so that it is not loaded for
    #  training and testing.
    return timeseries_input, timeseries_label


def load_dataset(filename, labelling_pattern=None, num_parallel_calls=None):
    if labelling_pattern:
        print(f"The dataset is loaded with the following labels:\n{labelling_pattern}")

    dataset = tf.data.TFRecordDataset(filename)
    # LINK: https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
    dataset = dataset.map(
        partial(read_tfrecord, labelling_pattern=labelling_pattern),
        num_parallel_calls=num_parallel_calls,
    )
    # NOTE: 'map' applies map_func to each element of this dataset, and returns a new dataset containing
    #  the transformed elements, in the same order as they appeared in the input. map_func can be used to
    #  change both the values and the structure of a dataset's elements.

    # Note: the code below might be useful if there are shape problems with tensors.
    # def get_data():
    #    for element in dataset:
    #        yield element

    # dataset =
    # tf.data.Dataset.from_generator(get_data, output_signature=tf.TensorSpec(shape=(1440, 36), dtype=tf.float32))
    return dataset


def get_dataset(
    filename,
    batch_size=64,
    shuffle_buffer_size=2048,
    prefetch_buffer_size=tf.data.AUTOTUNE,
    num_parallel_calls=tf.data.AUTOTUNE,
    shuffle=True,
    labelling_pattern=None,
):
    dataset = load_dataset(
        filename,
        labelling_pattern=labelling_pattern,
        num_parallel_calls=num_parallel_calls,
    )

    print('data from function :', dataset)

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)

    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    # NOTE: prefetch - Most dataset input pipelines should end with a call to prefetch.
    #  This allows later elements to be prepared while the current element is being processed.
    #  This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.

    return dataset


def get_dataset_dim(ds, has_label):
    # Todo: this function has to be adapted so that it can deal with any type of data structure.
    if has_label:
        timeseries_input, label = next(iter(ds))
    else:
        timeseries_input, label = next(iter(ds))

    shape = list(timeseries_input.get_shape())
    print("shape", shape)

    batch_size, segment_length, channels = shape

    return batch_size, segment_length, channels


def preprocess_data(dataset, preprocess, num_parallel_calls=tf.data.AUTOTUNE):
    if preprocess is not None:
        dataset = dataset.map(
            lambda x, y: (preprocess(x), y),
            num_parallel_calls=num_parallel_calls,
        )
    return dataset
