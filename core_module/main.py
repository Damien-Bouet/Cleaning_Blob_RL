import argparse
from peaceful_pie.unity_comms import UnityComms
import numpy as np

import os
from typing import List, Tuple, Callable, Optional
import pylab
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense,LeakyReLU ,Flatten, Input
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras import backend as K
import copy
from datetime import datetime

import time

from core_module.unity_env import MyEnv
from core_module.utils import str2bool
from core_module.fake_tensorboard import FakeTensorboard
from core_module.exceptions import UnityConnectionError

tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass


LOSS_CLIPPING = 0.2
ENTROPY_LOSS = 0.001
 
class ActorModel:
    """
    Actor Model for the PPO algorithm
    """
    def __init__(self, input_shape: Tuple[int, int, int], action_space: int, optimizer: Optimizer) -> None:
        """
        Initialize the Actor model for Proximal Policy Optimization.

        Parameters:
        - input_shape (Tuple[int, int, int]): The shape of the input state.
        - action_space (int): The dimension of the action space.
        - optimizer (Optimizer): The optimizer for model training.

        Returns:
        None
        """
        self.action_space = action_space

        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', strides=(1, 1), input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Flatten())
        model.add(Dense(self.action_space, activation="softmax"))

        self.Actor = model

        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer)
        
    def ppo_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Proximal Policy Optimization loss function.

        Parameters:
        - y_true (tf.Tensor): True labels (advantages, prediction_picks, actions).
        - y_pred (tf.Tensor): Predicted labels.

        Returns:
        total_loss (tf.Tensor): The total PPO loss.
        """
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Make predictions using the Actor model.

        Parameters:
        - state (np.ndarray): The input state.

        Returns:
        predictions (np.ndarray): The output predictions.
        """
        return self.Actor.predict(state/255)


class CriticModel:
    """
    Critic Model for the PPO algorithm
    """
    def __init__(self, input_shape: Tuple[int, int, int], optimizer: Optimizer) -> None:
        """
        Initialize the Critic model for Proximal Policy Optimization.

        Parameters:
        - input_shape (Tuple[int, int, int]): The shape of the input state.
        - optimizer (Optimizer): The optimizer for model training.

        Returns:
        None
        """
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))

        V = Conv2D(16, (3, 3), activation='relu', strides=(1, 1), input_shape=input_shape)(X_input)
        V = LeakyReLU(alpha=0.01)(V)
        V = Flatten()(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[X_input, old_values], outputs = value)

        self.Critic.compile(loss=[self.critic_ppo2_loss(old_values)], optimizer=optimizer)


    def critic_ppo2_loss(self, values: tf.Tensor) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
        """
        Define the PPO2 loss function for the Critic model.

        Parameters:
        - values (tf.Tensor): The old predicted values.

        Returns:
        loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): The PPO2 loss function.
        """
        def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            """
            PPO2 loss function for the Critic model.

            Parameters:
            - y_true (tf.Tensor): True labels.
            - y_pred (tf.Tensor): Predicted labels.

            Returns:
            value_loss (tf.Tensor): The value loss.
            """

            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            return value_loss
        
        return loss

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Make predictions using the Critic model.

        Parameters:
        - state (np.ndarray): The input state.

        Returns:
        predictions (np.ndarray): The output predictions.
        """
        return self.Critic.predict([state/255, np.zeros((state.shape[0], 1))])


class PPOAgent:
    """
    Agent, trained with the RL PPO algorithm
    """
    # PPO Main Optimization Algorithm
    def __init__(
        self,
        env: MyEnv,
        total_episodes: int = 10000,
        training_batch: int = 200,
        log_dir: Optional[str] = None,
        load_model_dir: Optional[str] = None,
        save_model_dir: Optional[str] = None,
        load_model: bool = False,
        save_model: bool = True,
        lr: float = 0.0005,
    ) -> None:
        """
        Initialize the PPOAgent.

        Parameters:
        - env (MyEnv): The environment.
        - total_episodes (int): Total episodes to train through all environments.
        - training_batch (int): Number of episodes to train in each batch.
        - log_dir (Optional[str]): The directory to save the tensorboard logs.
        - load_model_dir (Optional[str]): The directory to load the pre-trained model.
        - save_model_dir (Optional[str]): The directory to save the trained model.
        - load_model (bool): Whether to load the pre-trained model.
        - save_model (bool): Whether to save the trained model.
        - lr (float): Learning rate.

        Returns:
        None
        """

        self.env = env
        self.env_name = "cleaning_blob"
        self.action_size : int = self.env.action_space.n
        self.state_size : Tuple[int, int, int] = self.env.observation_space.shape
        self.total_episodes = total_episodes # total episodes to train through all environments
        self.episode :int = 0 # used to track the episodes total count of episodes played through all thread environments
        self.step :int = 0
        self.previous_step :int = 0
        self.max_average : float = 0. # when average score is above 0 model will be saved
        self.lr = lr
        self.episode_late_save : int = 0
        self.shift_average : int = 25 #average over last 25 episodes (to save the model if average improves)
        self.decay_lr_plateau : int = 50 #reduce lr if averaged results didn't improved over this nbr of episode
        self.epochs : int = 10 # training epochs
        self.shuffle : bool =False
        self.training_batch = training_batch
        #self.optimizer = RMSprop
        self.optimizer : Optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)#Adam(lr=lr)

        self.replay_count : int = 0

         # Instantiate plot memory
        self.scores_ : List[float] = []
        self.episodes_ : List[int] = []
        self.average_ : List[float] = []


        if(log_dir is None):
            self.log_dir = "logs/train/"+ str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S/"))
            self.tensorboard = FakeTensorboard(self.log_dir)
        else:
            self.log_dir = log_dir
            self.tensorboard = FakeTensorboard(self.log_dir)
            self.tensorboard.load_csv()
            self.episode = len(self.tensorboard.data['reward']) +1
            self.episode_late_save = self.episode
            self.step = len(self.tensorboard.data['Step per second']) +1
            self.previous_step = self.step

            self.replay_count = len(self.tensorboard.data['Actor loss per replay']) +1

            rewards = np.array(self.tensorboard.data['reward'])[:,0]
            for x in rewards:
                self.scores_.append(x)
                self.average_.append(sum(self.scores_[-self.shift_average:]) / len(self.scores_[-self.shift_average:]))
            self.max_average = max(self.average_)
        
        if(load_model_dir is None):
            self.load_model_dir = "models/"+ str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S/"))
        else:
            self.load_model_dir = load_model_dir
        if(save_model_dir is None):
            self.save_model_dir = "models/"+ str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S/"))
        else:
            self.save_model_dir = save_model_dir

        if(self.load_model_dir[-1] != "/"):
            self.load_model_dir += "/"
        if(self.save_model_dir[-1] != "/"):
            self.save_model_dir += "/"

        if not os.path.exists(self.load_model_dir):
            os.makedirs(self.load_model_dir)
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)

        # Create Actor-Critic network models
        self.actor = ActorModel(input_shape=self.state_size, action_space = self.action_size, optimizer = self.optimizer)
        self.critic = CriticModel(input_shape=self.state_size, optimizer = self.optimizer)

        self.actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.critic_name = f"{self.env_name}_PPO_Critic.h5"

        if(load_model):
            self.load()

        self.save_model = save_model


    def act(self, state: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
        """
        Choose an action.

        Parameters:
        - state (np.ndarray): The current state.

        Returns:
        Tuple[int, np.ndarray, np.ndarray]: Action, one-hot encoded action, and prediction.
        """
        # Use the network to predict the next action to take, using the model
        prediction = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction


    def discount_rewards(self, reward: np.ndarray) -> np.ndarray:
        """
        Compute the gamma-discounted rewards over an episode.
        Consider that it is generally better to use the Generalized Advantage Estimation (get_gaes function).

        Parameters:
        - reward (np.ndarray): The rewards.

        Returns:
        np.ndarray: Discounted rewards.
        """
        # get_gaes is better
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        return discounted_r


    def get_gaes(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        gamma: float = 0.99,
        lamda: float = 0.9,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Parameters:
        - rewards (np.ndarray): The rewards.
        - dones (np.ndarray): The done flags.
        - values (np.ndarray): Current state values.
        - next_values (np.ndarray): Next state values.
        - gamma (float): Discount factor.
        - lamda (float): Lambda parameter.
        - normalize (bool): Whether to normalize GAE.

        Returns:
        Tuple[np.ndarray, np.ndarray]: Advantages and target values.
        """
        deltas = np.array([r + gamma * (1 - d) * nv - v for (r, d, nv, v) in zip(rewards, dones, next_values, values)])
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)


    def replay(
        self,
        states: np.ndarray,
        actions: List[np.ndarray],
        rewards: List[float],
        predictions: List[np.ndarray],
        dones: List[bool],
        next_states: np.ndarray,
    ) -> None:
        """
        Replay the experiences and train the networks.

        Parameters:
        - states (np.ndarray): Current states.
        - actions (List[np.ndarray]): List of actions.
        - rewards (List[float]): List of rewards.
        - predictions (List[np.ndarray]): List of predictions.
        - dones (List[bool]): List of done flags.
        - next_states (np.ndarray): Next states.

        Returns:
        None
        """
        # reshape memory to appropriate shape for training
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.critic.predict(states)
        next_values = self.critic.predict(next_states)

        # Compute discounted rewards and advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
       
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, predictions, actions])

        # training Actor and Critic networks
        a_loss = self.actor.Actor.fit(states/255, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.critic.Critic.fit([states/255, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        self.tensorboard.add_scalar("Actor loss per replay", np.sum(a_loss.history['loss']), step=self.replay_count)    
        self.tensorboard.add_scalar("Critic loss per replay", np.sum(c_loss.history['loss']), step=self.replay_count)

        self.replay_count += 1


    def load(self) -> None:
        """
        Load the pre-trained model weights.

        Returns:
        None
        """
        self.actor.Actor.load_weights(self.load_model_dir + self.actor_name)
        self.critic.Critic.load_weights(self.load_model_dir + self.critic_name)


    def save(self, suffix: str = "") -> None:
        """
        Save the trained model weights.

        Parameters:
        - suffix (str): Suffix for the saved model file.

        Returns:
        None
        """
        self.actor.Actor.save_weights(self.save_model_dir + self.actor_name[:-3] + suffix + ".h5")
        self.critic.Critic.save_weights(self.save_model_dir + self.critic_name[:-3]+ suffix + ".h5")
        
    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    
    
    def plot_model(self, score: float, episode: int) -> Tuple[float, str]:
        """
        Plot the training progress and update learning rate.

        Parameters:
        - score (float): The current score.
        - episode (int): The current episode.

        Returns:
        Tuple[float, str]: Current average and saving information.
        """
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-self.shift_average:]) / len(self.scores_[-self.shift_average:]))
        if (str(episode)[-2:] == "00") or (str(episode)[-2:] == "50"):# much faster than episode % 100
            if os.path.exists(str(self.log_dir)+"out.csv"):
                os.remove(str(self.log_dir)+"out.csv")
            self.tensorboard.save_to_csv(self.log_dir)
            self.save("_"+str(episode))

        # saving best models
        if self.average_[-1] >= self.max_average and self.episode > self.shift_average:
            self.max_average = self.average_[-1]
            if(self.save_model):
                self.save()
                saving = "-----------------SAVING-----------------"
            
            # decreaate learning rate every saved model
            # self.lr *= 0.95
            # K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            # K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)

            self.episode_late_save = self.episode
        else:
            saving = ""
            

        if(self.episode - self.episode_late_save > self.decay_lr_plateau):
            self.episode_late_save = self.episode
            self.lr *= 0.5
            K.set_value(self.actor.Actor.optimizer.learning_rate, self.lr)
            K.set_value(self.critic.Critic.optimizer.learning_rate, self.lr)

        return self.average_[-1], saving
    
    
    def run_batch(self) -> None:
        """
        Run a batch of episodes for training.

        Returns:
        None
        """
        # train every self.Training_batch episodes
        state = self.env.reset()[0]
        printing_period = 1
        
        previous_time = time.time()
        done, score, saving = False, 0, ''
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, episode_steps, predictions, dones = np.zeros((self.training_batch,24,32,1)), np.zeros((self.training_batch,24,32,1)), [], [], [], [], []
            for t in range(self.training_batch):
                # self.env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act(np.expand_dims(state, axis=0))
                # Retrieve new state, reward, and whether the state is terminal
                
                next_state, reward, done, _, _ = self.env.step(action)
                self.step += 1

                self.tensorboard.add_scalar("Step per second", 1/(time.time() - previous_time), step=self.step)
                previous_time = time.time()
                
                # Memorize (state, action, reward) for training
                states[t,:,:,:] = state
                
                next_states[t,:,:,:] = next_state
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = next_state
                score += reward
                if done:
                    episode_steps.append(self.step-self.previous_step)
                    self.previous_step = self.step
                    
                    self.tensorboard.add_scalar("reward", score, step=self.episode)
                    self.tensorboard.add_scalar("episode steps", episode_steps[-1], step=self.episode)
                    self.tensorboard.add_scalar("learning rate", self.lr, step=self.episode)
                    # dqn_variable = reward
                    # tf.summary.histogram(name="agent_rewards", data=tf.convert_to_tensor(dqn_variable), step=step)
                        

                    self.episode += 1
                    average, saving = self.plot_model(score, self.episode)
                    if(self.episode % printing_period == 0):
                        print(f"episode: {self.episode}/{self.total_episodes}, step: {self.step}, average episode step: {np.mean(episode_steps[-50:])} score: {score}, average: {round(average,2)} {saving}")

                    state, done, score, saving = self.env.reset()[0], False, 0, ''
                
            self.replay(states, actions, rewards, predictions, dones, next_states)
            if self.episode >= self.total_episodes:
                break
        
        self.tensorboard.save_to_csv()
        print("------------------------------------------------------------------------")
        print(f"Tensorboard saved, create it by running : python create_tensorboard.py --path {str(self.log_dir)}")
        print("-----------------------------------------------------------------------")
        # self.env.close()  


    def test(self, test_episodes: int = 100) -> None:
        """
        Test the trained agent.

        Parameters:
        - test_episodes (int): Number of episodes for testing.

        Returns:
        None
        """
        self.load()
        for e in range(test_episodes):
            state = self.env.reset()[0]
            
            done = False
            score = 0
            while not done:
                # self.env.render()
                prediction = self.actor.predict(np.expand_dims(state, axis=0))[0]
                action = np.random.choice(self.action_size, p=prediction)
                state, reward, done, _, _ = self.env.step(action)
                # state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    print(f"episode: {e}/{test_episodes}, score: {score}")
                    break
        self.env.close()


def run(args: argparse.Namespace) -> None:
    """
    Run the training or testing based on the provided arguments.

    Parameters:
    - args (argparse.Namespace): Command-line arguments.

    Returns:
    None
    """

    if(args.list_models_directories):
        folder = './models/'
        for name in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, name)):
                print(name)                             

        return
    
    unity_comms = UnityComms(port = args.port)
    if unity_comms.Say(message=args.message, retry=False) is not None:
        print("The connection is working. The training will start.")
    else:
        raise UnityConnectionError()
    
    my_env = MyEnv(unity_comms=unity_comms)
    agent = PPOAgent(
        my_env,
        total_episodes=args.episodes,
        training_batch=args.training_batch,
        log_dir=args.log_dir,
        load_model_dir=args.load_model_dir,
        save_model_dir=args.save_model_dir,
        load_model=args.load_model,
        save_model=args.save_model,
        lr=args.lr
    )
    if(args.test):
        agent.test()
    elif(args.train):
        agent.run_batch()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=9000, help='Port to connect to Unity - default:9000')

    parser.add_argument("--list_models_directories", type=str2bool, nargs='?', const=True, default=False, help="Print the models directories to select easily the model you want to load. If True, don't train, just print the subdirectories - default:False")

    parser.add_argument("--train", type=str2bool, nargs='?', const=True, default=True, help="Train mode - default:True")
    parser.add_argument("--test", type=str2bool, nargs='?', const=True, default=False, help="Test mode - default:False")

    parser.add_argument("--load_model_dir", type=str, default=None, help="Load previous model at the directory - default: None => models/dd-mm-YYYY_hh-mm-ss")
    parser.add_argument("--save_model_dir", type=str, default=None, help="Save previous model at the directory - default: None => models/dd-mm-YYYY_hh-mm-ss")
    parser.add_argument("--load_model", type=str2bool, nargs='?', const=True, default=False, help="Load the model during training - default:False")
    parser.add_argument("--save_model", type=str2bool, nargs='?', const=True, default=True, help="Save the model during training - default:True")

    parser.add_argument('--log_dir', type=str, default=None, help='logs directory, to resume a previous training - default:None => logs/train/dd-mm-YYYY_hh-mm-ss')
    parser.add_argument('--episodes', type=int, default=10000, help='Training episodes number - default:10000')
    parser.add_argument('--training_batch', type=int, default=200, help='Training batch size - default:200')
    parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate, interesting to change it if we want to resume a previous training of a model with the previous late learning rate - default:0.0005')
    args = parser.parse_args()

    run(args)

