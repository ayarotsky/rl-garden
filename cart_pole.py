import tensorflow as tf
import imageio
from tqdm import tqdm

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

num_iterations = 40000
initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3
num_eval_episodes = 10
env_name = 'CartPole-v0'

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

q_net = q_network.QNetwork(train_env.observation_spec(),
                           train_env.action_spec(),
                           fc_layer_params=(100,))
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
agent = dqn_agent.DqnAgent(train_env.time_step_spec(),
                           train_env.action_spec(),
                           q_network=q_net,
                           optimizer=optimizer,
                           td_errors_loss_fn=common.element_wise_squared_loss,
                           train_step_counter=tf.Variable(0))

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics
def compute_avg_return(environment, policy, num_episodes=10):
  total_return = 0.0

  for _ in range(num_episodes):
    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward

    total_return += episode_return

  avg_return = total_return / num_episodes

  return avg_return.numpy()[0]

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

# https://github.com/tensorflow/agents/blob/master/tf_agents/docs/python/tf_agents/drivers.md
def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env,
             random_policy,
             replay_buffer,
             steps=initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

iterator = iter(dataset)

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

for _ in tqdm(range(num_iterations), desc='Training DQN'):
  # Collect a few steps using collect_policy and save to the replay buffer.
  for _ in range(collect_steps_per_iteration):
    collect_step(train_env, agent.collect_policy, replay_buffer)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  agent.train(experience)

def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
  with imageio.get_writer(f'{filename}.mp4', fps=fps) as video:
    for _ in tqdm(range(num_episodes), desc='Evaluating DQN'):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())

      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())

create_policy_eval_video(agent.policy, 'trained-agent', num_eval_episodes)
