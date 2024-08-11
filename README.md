
# Introduction 

The goal is to automate a robot that cleans boat hulls. The robot is held against the hull by a suction system, and is remotely controlled by an operator, using video feedback. The aim is to use a reinforcement learning algorithm to automate this task, making the robot clean the hull autonomously, thanks only to the video feedback. We will simulate the environment in Unity 3D, and train the agent using PPO algorithm in Python.

As we can see just below, we simplified the boat hull with a half-capsule and crated many green squares on its surface to simulate the algae. The agent recieves at each step, the observation of the camera sensor, and the reward of the previous step, corresponding to the number of algae it cleaned. We clearly see that the agent learned by itself to clean the boat in an optimaized way, with a circular motion. (The agent is placed in a random position at the beginning, so it didn't just memorized the path, but really learned to analyse the visiual input). Tuning the reward function and discount factor to future rewards is crucial to successful RL. Due to the suction mechanism not working outside the water, we ended the episode and incurred a large negative reward if the robot drove out of the water. We then added slighly larger rewards for algae tiles close to the surface to ensure that the robot was motivated not to be overly catious around this danger (and thus miss portions of the algae). We also faced a problem with the robot underutilizing turning (a relatively slow action to perform on the physical robot). Turning yielded no rewards since no algae could not be collected. In some training runs, the robot exhibited the behavior of shuffling back and forth in quick succession (with miniscule turns, or not at all, taking advantage of the curved surface to adjust its trajectory). This behavior resulted in rapidly cleaning a large section but leaving slivers of missed spots and driving blindly in reverse naturally increases the chances of inadvertently surfacing the robot. We rectified this problem by decreasing the discount factor and increasing the episode length. This both discouraged leaving discountinous patches of uncollected algae that would be difficult to collect later and decreased the short-term greediness of the reward system. Interestingly, the obvious solution to promote turning of giving a slight postive reward for taking that action worsened our problem, since then the robot would then exhibit similar short-sighted behavior and rather than rectify its missed spots later in the episode, would simply give up and spin in place. 

![This is a gif](assets/cleaning_blob_demo.gif)

---
# Getting Started

To use the project on your own:

* Clone the repo on your disk

* Download Unity 3D (Tested on Unity 2021.3.21f1)

* Open a terminal in the project root folder (Cleaning_Blob_RL)

* Create a new conda virtual environment associated to the project:
```bash
$ conda create -y --name <project-env> python=3.8
$ conda activate <project-env>
```

(you may have to use conda init before being able to run this command),

* run:
```bash
$ pip install -r requirements.txt
```

---
# Python-Unity connection:

* Open the Unity Hub
* Click on ``Add project from disk``
* Open the project in the folder ``Cleaning_Blob_RL/Unity_files``
* Start the Scene with the Play button.

The connection should automatically be opened 


## Send a message to Unity
* Run the following command in the terminal:

```bash
$ python -m core_module.test_unity_connection.say
```
You should see in the Unity Logs a message (default to "Hello World"). If you can indeed see the message, your connection is working. Stop the Unity Scene to close the connection.
You can change the localhost connection port in the Unity interface, in the Agent script parameters (default to 9000)

## Play with the other functions
You can play with the different functions, in particular:

```bash
$ python -m core_module.test_unity_connection.reset
$ python -m core_module.test_unity_connection.step --action 1 --show_observation
```
This will reset the environnement, generate the algae, and ask for the agent to step forward. You will also recieve the observation back.

---
# Train the Agent

To train the Agent, you should first open the connection between python and Unity. See the above ``Python-Unity connection`` section for more details.

To train a new agent from scratch, run:
```bash
$ python -m core_module.main --train
```

You can see all your saved models (models are saved by default) using the command :
```bash
$ python -m core_module.main --list_models_directories
```

To test you Agent, load a previously trained model:
```bash
$ python -m core_module.main --test --load_model --load_model_dir "your trained model directory"
```

To resume training, load the model with which you wish to continue training. It may also be important to set the learning rate to the value it had at the end of the previous learning stage.
```bash
$ python -m core_module.main --train --load_model --load_model_dir "previous_model_dir" --log_dir "previous_model_logs_dir" --lr last_lr_value
```

You can see all the possible parameter of the main script using 
```bash
$ python -m core_module.main --help
```

---
# Last words
Don't hesitate to change directly in the script the model parameters and optimizer to find the best configuration. And most importantly, have fun !
