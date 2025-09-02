from models.templates.climate_processing_LSTM_CNN import climate_processing_LSTM_CNN
from models.templates.gapfilling_autoencoder2D import gapfilling_autoencoder2D
from models.templates.gapfilling_LSTM_CNN import gapfilling_LSTM_CNN
from models.templates.gapfilling_LSTM_encoder import gapfilling_LSTM_encoder


def get_model(model, **kwargs):
    if model == "climate_processing_LSTM_CNN":
        return climate_processing_LSTM_CNN(**kwargs)
    elif model == "gapfilling_autoencoder2D":
        return gapfilling_autoencoder2D(**kwargs)
    elif model == "gapfilling_LSTM_CNN":
        return gapfilling_LSTM_CNN(**kwargs)
    elif model == "gapfilling_LSTM_encoder":
        return gapfilling_LSTM_encoder(**kwargs)
    else:
        raise ValueError(f"Unknown model_name {model}")
