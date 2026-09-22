import tensorflow as tf

from src.exceptions import ConfigurationError


def build_lstm(
    window_size: int,
    feature_count: int,
    units: int,
    learning_rate: float,
) -> tf.keras.Model:
    if min(window_size, feature_count, units) < 1:
        raise ConfigurationError("Model dimensions must be positive.")

    if learning_rate <= 0:
        raise ConfigurationError("Learning rate must be positive.")

    model = tf.keras.Sequential(
        [
            tf.keras.Input(
                shape=(window_size, feature_count),
                name="price_history",
            ),
            tf.keras.layers.LSTM(
                units=units,
                return_sequences=False,
                stateful=False,
                name="lstm",
            ),
            tf.keras.layers.Dense(
                units=1,
                activation="linear",
                name="next_close",
            ),
        ],
        name="close_lstm",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae")
        ],
    )

    return model