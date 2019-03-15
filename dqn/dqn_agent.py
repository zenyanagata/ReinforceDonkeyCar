import os
import random
import argparse
import sys
import signal

from collections import deque
import numpy as np
from tensorflow.python import keras as K
from PIL import Image
import gym
import gym_ple
from fn_framework import FNAgent, Trainer, Observer
import tensorflow as tf

import donkey_gym



class DeepQNetworkAgent(FNAgent):
    """
    DQNではQ tableの代わりにstate, actionを入力し、各行動に対するQ値を出力する
    Neural Networkが用いられる。
    """

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._scaler = None
        self._teacher_model = None

    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.model.compile(optimizer, loss="mse")
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(K.layers.Conv2D(
            24, kernel_size=5, strides=2, padding="same",
            input_shape=feature_shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            32, kernel_size=5, strides=2, padding="same",
            input_shape=feature_shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=5, strides=2, padding="same",
            input_shape=feature_shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=3, strides=2, padding="same",
            input_shape=feature_shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=4, strides=2, padding="same",
            kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Conv2D(
            64, kernel_size=3, strides=1, padding="same",
            input_shape=feature_shape, kernel_initializer=normal,
            activation="relu"))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(512, kernel_initializer=normal,
                                 activation="relu"))
        model.add(K.layers.Dense(15, kernel_initializer=normal,
                                 activation="linear"))
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)

    def linear_unbin(self, arr):
        """
        Convert a categorical array to value.
        長さが15の配列（angle -1 ~ 1を15当分したときの各angleのQ値が入っている）を
        受け取り、-1 ~ 1の値に変換して返す。
        例： arr = [1.0, 0.7, 0.3, 0.0, 0.0, ..., 0.0] -> a = -1.0
             arr = [0.2, 0.3, 0.4, 0.5, 0.4, 0.3, ....] -> a = -0.63

        See Also
        --------
        linear_bin
        """
        if not len(arr) == 15:
            raise ValueError('Illegal array length, must be 15')
        b = np.argmax(arr)
        a = b * (2 / 14) - 1
        return a

    # 例のget_actionと一致させている
    def policy(self, s, env):
        if np.random.random() < self.epsilon or not self.initialized:
            return env.action_space.sample()[0]
        else:
            estimates = self.estimate(s)  # estimated q value [q_angle, q_throttle]
            return self.linear_unbin(estimates[0])        

    def estimate(self, state):
        return self.model.predict(np.array([state]))  # [angle, throttle]
    
    def update(self, experiences, gamma):
        states = np.array([e.s for e in experiences])
        n_states = np.array([e.n_s for e in experiences])

        estimateds = self.model.predict(states)
        future = self._teacher_model.predict(n_states)

        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward

        loss = self.model.train_on_batch(states, estimateds)
        return loss

    def update_teacher(self):
        self._teacher_model.set_weights(self.model.get_weights())


class CatcherObserver(Observer):

    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen=frame_count)

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0~1
        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)
        feature = np.array(self._frames)
        # Convert the feature shape (f, w, h) => (w, h, f).
        feature = np.transpose(feature, (1, 2, 0))

        return feature


class DeepQNetworkTrainer(Trainer):

    def __init__(self, buffer_size=50000, batch_size=32,
                 gamma=0.99, initial_epsilon=0.5, final_epsilon=1e-3,
                 learning_rate=1e-3, teacher_update_freq=3, report_interval=10,
                 log_dir="", file_name=""):
        super().__init__(buffer_size, batch_size, gamma,
                         report_interval, log_dir)
        self.file_name = file_name if file_name else "dqn_agent.h5"
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.learning_rate = learning_rate
        self.teacher_update_freq = teacher_update_freq
        self.loss = 0
        self.training_episode = 0
        self._max_reward = -10

    def train(self, env, episode_count=1200, initial_count=200,
              render=False, observe_interval=100):
        actions = list(range(2))  # Steering and Throttle
        agent = DeepQNetworkAgent(1.0, actions)  # epsilon初期値, actions

        self.training_episode = episode_count

        self.train_loop(env, agent, episode_count, initial_count, render,
                        observe_interval)
        return agent

    def episode_begin(self, episode, agent):
        self.loss = 0

    def begin_train(self, episode, agent):
        optimizer = K.optimizers.Adam(lr=self.learning_rate, clipvalue=1.0)
        agent.initialize(self.experiences, optimizer)
        self.logger.set_model(agent.model)
        agent.epsilon = self.initial_epsilon
        self.training_episode -= episode

    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            self.loss += agent.update(batch, self.gamma)

    def episode_end(self, episode, step_count, agent):
        reward = sum([e.r for e in self.get_recent(step_count)])
        self.loss = self.loss / step_count
        self.reward_log.append(reward)
        if self.training:
            self.logger.write(self.training_count, "loss", self.loss)
            self.logger.write(self.training_count, "reward", reward)
            self.logger.write(self.training_count, "epsilon", agent.epsilon)
            if reward > self._max_reward:
                agent.save(self.logger.path_of(self.file_name))
                self._max_reward = reward
            if self.is_event(self.training_count, self.teacher_update_freq):
                agent.update_teacher()

            diff = (self.initial_epsilon - self.final_epsilon)
            decay = diff / self.training_episode
            agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)

        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval:]
            self.logger.describe("reward", recent_rewards, episode=episode)


def main(args):
    EPISODES = 10000
    img_rows , img_cols = 80, 80
    img_channels = 4 # We stack 4 frames

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    #we pass arguments to the donkey_gym init via these
    os.environ['DONKEY_SIM_PATH'] = args.sim
    os.environ['DONKEY_SIM_PORT'] = str(args.port)
    os.environ['DONKEY_SIM_HEADLESS'] = str(args.headless)

    env = gym.make(args.env_name)

    #not working on windows...
    def signal_handler(signal, frame):
        print("catching ctrl+c")
        env.unwrapped.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGABRT, signal_handler)


    file_name = "dqn_agent.h5"
    trainer = DeepQNetworkTrainer(file_name=file_name, buffer_size=10000, batch_size=64,
                 gamma=0.99, initial_epsilon=1.0, final_epsilon=2e-2,
                 learning_rate=1e-4, teacher_update_freq=3, report_interval=10,
                 log_dir="")
    path = trainer.logger.path_of(trainer.file_name)
    agent_class = DeepQNetworkAgent

    obs = CatcherObserver(env, img_rows, img_cols, img_channels)

    if args.play:
        print("Loading pretrained model in {}.".format(path))
        agent = agent_class.load(obs, path)
        agent.play(obs, render=True)
    else:
        trainer.train(obs, episode_count=EPISODES, 
                    initial_count=10, render=False, observe_interval=100)


if __name__ == "__main__":    
    # Initialize the donkey environment
    # where env_name one of:    
    env_list = [
       "donkey-warehouse-v0",
       "donkey-generated-roads-v0",
       "donkey-avc-sparkfun-v0",
       "donkey-generated-track-v0"
    ]

    parser = argparse.ArgumentParser(description='ddqn')
    parser.add_argument('--sim', type=str, default="manual", help='path to unity simulator. maybe be left at manual if you would like to start the sim on your own.')
    parser.add_argument('--model', type=str, default="rl_driver.h5", help='path to model')
    parser.add_argument('--test', action="store_true", help='agent uses learned model to navigate env')
    parser.add_argument('--headless', type=int, default=0, help='1 to supress graphics')
    parser.add_argument('--port', type=int, default=9091, help='port to use for websockets')
    parser.add_argument('--throttle', type=float, default=0.3, help='constant throttle for driving')
    parser.add_argument('--env_name', type=str, default='donkey-generated-track-v0', help='name of donkey sim environment', choices=env_list)
    parser.add_argument("--play", action="store_true", help="play with trained model")

    args = parser.parse_args()

    main(args)
