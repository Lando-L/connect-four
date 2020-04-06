from typing import List, Tuple

import numpy as np
from tensorflow import keras

from connectfour.game.gamestate import GameState


class AgentModel:
    def __init__(self, observation_space: Tuple, action_space: int, value_space: int):
        self.observation_space = observation_space
        self.action_space = action_space
        self.value_space = value_space

        self.model = None

    @staticmethod
    def __state_to_input(state: GameState) -> np.ndarray:
        """
        Returns the transformed state according to the model's input.

        :param state: GameState
        :return: Transformed state
        """

        return np.insert(state.board, 2, np.full(state.board.shape[:2], state.player), axis=-1)

    def build(self, learning_rate: float, dropout: float) -> None:
        """
        Builds a new model.

        :param learning_rate: Learning rate
        :param dropout: Dropout rate
        """

        def convolution_block(inputs, prefix: str, num_filters: int):
            tower_1_conv_1 = keras.layers.Conv2D(num_filters, (1, 1), padding='same', name=f'{prefix}_tower_1_convolution_layer_1')(inputs)
            tower_1_conv_1_normalisation = keras.layers.BatchNormalization(name=f'{prefix}_tower_1_normalisation_layer_1')(tower_1_conv_1)
            tower_1_conv_1_activation = keras.layers.Activation('relu', name=f'{prefix}_tower_1_activation_layer_1')(tower_1_conv_1_normalisation)

            tower_2_conv_1 = keras.layers.Conv2D(num_filters, (1, 1), padding='same', name=f'{prefix}_tower_2_convolution_layer_1')(inputs)
            tower_2_conv_1_normalisation = keras.layers.BatchNormalization(name=f'{prefix}_tower_2_normalisation_layer_1')(tower_2_conv_1)
            tower_2_conv_1_activation = keras.layers.Activation('relu', name=f'{prefix}_tower_2_activation_layer_1')(tower_2_conv_1_normalisation)
            tower_2_conv_2 = keras.layers.Conv2D(num_filters, (3, 3), padding='same',name=f'{prefix}_tower_2_convolution_layer_2')(tower_2_conv_1_activation)
            tower_2_conv_2_normalisation = keras.layers.BatchNormalization(name=f'{prefix}_tower_2_normalisation_layer_2')(tower_2_conv_2)
            tower_2_conv_2_activation = keras.layers.Activation('relu', name=f'{prefix}_tower_2_activation_layer_2')(tower_2_conv_2_normalisation)

            tower_3_conv_1 = keras.layers.Conv2D(num_filters, (1, 1), padding='same',name=f'{prefix}_tower_3_convolution_layer_1')(inputs)
            tower_3_conv_1_normalisation = keras.layers.BatchNormalization(name=f'{prefix}_tower_3_normalisation_layer_1')(tower_3_conv_1)
            tower_3_conv_1_activation = keras.layers.Activation('relu', name=f'{prefix}_tower_3_activation_layer_1')(tower_3_conv_1_normalisation)
            tower_3_conv_2 = keras.layers.Conv2D(num_filters, (5, 5), padding='same', name=f'{prefix}_tower_3_convolution_layer_2')(tower_3_conv_1_activation)
            tower_3_conv_2_normalisation = keras.layers.BatchNormalization(name=f'{prefix}_tower_3_normalisation_layer_2')(tower_3_conv_2)
            tower_3_conv_2_activation = keras.layers.Activation('relu', name=f'{prefix}_tower_3_activation_layer_2')(tower_3_conv_2_normalisation)

            tower_4_pool = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=f'{prefix}_tower_4_pooling_layer')(inputs)
            tower_4_conv_1 = keras.layers.Conv2D(num_filters, (1, 1), padding='same', name=f'{prefix}_tower_4_convolution_layer_1')(tower_4_pool)
            tower_4_conv_1_normalisation = keras.layers.BatchNormalization(name=f'{prefix}_tower_4_normalisation_layer_1')(tower_4_conv_1)
            tower_4_conv_1_activation = keras.layers.Activation('relu', name=f'{prefix}_tower_4_activation_layer_1')(tower_4_conv_1_normalisation)

            concatenation = keras.layers.Concatenate(name=f'{prefix}_concatenation')([
                tower_1_conv_1_activation,
                tower_2_conv_2_activation,
                tower_3_conv_2_activation,
                tower_4_conv_1_activation
            ])

            return concatenation

        def dense_block(inputs, prefix: str, num_hidden_units: int, dropout: float):
            dense = keras.layers.Dense(num_hidden_units, name=f'{prefix}_dense_layer_1')(inputs)
            normalisation = keras.layers.BatchNormalization(name=f'{prefix}_batch_normalisation_layer_1')(dense)
            activation = keras.layers.Activation('relu', name=f'{prefix}_activation_layer_1')(normalisation)
            dropout = keras.layers.Dropout(dropout, name=f'{prefix}_dropout_layer')(activation)

            return dropout

        inputs = keras.layers.Input(self.observation_space, name='input_layer')

        convolution_block_1 = convolution_block(inputs, 'inception_block_1', 32)
        convolution_block_2 = convolution_block(convolution_block_1, 'inception_block_2', 32)
        convolution_block_3 = convolution_block(convolution_block_2, 'inception_block_3', 32)
        convolution_block_4 = convolution_block(convolution_block_3, 'inception_block_4', 32)
        convolution_block_5 = convolution_block(convolution_block_4, 'inception_block_5', 32)

        pooling = keras.layers.AveragePooling2D(pool_size=(3, 3))(convolution_block_5)
        flatten = keras.layers.Flatten(name='flatten_layer')(pooling)

        policy_block = dense_block(flatten, 'policy_block', 64, dropout)
        policy_logits = keras.layers.Dense(self.action_space, name='policy_logits_layer')(policy_block)
        policy_output = keras.layers.Activation('sigmoid', name='policy_output')(policy_logits)

        value_block = dense_block(flatten, 'value_block', 64, dropout)
        value_logits = keras.layers.Dense(self.value_space, name='value_logits_layer')(value_block)
        value_output = keras.layers.Activation('softmax', name='value_output')(value_logits)

        self.model = keras.models.Model(inputs=inputs, outputs=[policy_output, value_output])
        self.model.compile(
            loss={'policy_output': 'mse', 'value_output': 'categorical_crossentropy'},
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics={'policy_output': 'mse', 'value_output': 'acc'}
        )

    def load(self, path: str) -> None:
        """
        Loads a model.

        :param path: File path
        """

        self.model = keras.models.load_model(path)

    def save(self, path: str) -> None:
        """
        Saves the current model.

        :param path: File path
        """

        self.model.save(path)

    def train(self,
              states: List[GameState],
              actions: List[np.ndarray],
              outcomes: List[int],
              batch_size: int,
              num_epochs: int) -> keras.callbacks.History:
        """
        Trains the model on sampled data.

        :param states: Sampled states
        :param actions: Sampled actions
        :param outcomes: Sampled outcomes
        :param batch_size: Batch size
        :param num_epochs: Number of epochs
        """

        return self.model.fit(
            x=np.stack(list(map(AgentModel.__state_to_input, states))),
            y={
                'policy_output': np.array(actions),
                'value_output': keras.utils.to_categorical(np.array(outcomes) + 1, num_classes=self.value_space)
            },
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=0
        )

    def evaluate(self, state: GameState) -> Tuple[np.ndarray, float]:
        """
        Returns the estimated value of the given state and probabilities of actions to take.

        :param state: GameState
        :return: Action probabilities and state value
        """

        labels = np.array([-1, 0, 1])
        prediction = self.model.predict(
            AgentModel.__state_to_input(state)[np.newaxis, ...],
            batch_size=1
        )

        probabilities = prediction[0].flatten()
        value = prediction[1].flatten().dot(labels)

        return probabilities, value
