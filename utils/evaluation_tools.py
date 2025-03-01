import matplotlib.pyplot as plt


def plot(path, model_id, y_predict, y_true, batch, sequence):

    plt.plot(y_predict[sequence], label='prediction')
    plt.plot(y_true[sequence], label='ground truth')
    plt.xlabel('time [days]')
    plt.legend()
    plt.savefig(path + '/figures/batch_' + str(batch) + '_sequence_' + str(sequence) + '_' + model_id + '.png')
    plt.close()
