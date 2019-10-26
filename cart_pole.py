import tensorflow as tf
import imageio
from tqdm import tqdm

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.drivers import dynamic_step_driver
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

def create_policy_eval_video(policy, env_name, num_episodes=5, fps=30):
    eval_py_env = suite_gym.load(env_name)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    with imageio.get_writer(f'{env_name}-DQN.mp4', fps=fps) as video:
        for _ in tqdm(range(num_episodes), desc='Evaluating DQN'):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())

num_iterations = 30000
initial_collect_steps = 1000
collect_steps_per_iteration = 1
replay_buffer_max_length = 100000
batch_size = 64
learning_rate = 1e-3
num_eval_episodes = 5
env_name = 'CartPole-v0'

train_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

q_net = q_network.QNetwork(train_env.observation_spec(),
                           train_env.action_spec(),
                           fc_layer_params=(100,))
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
    td_errors_loss_fn=common.element_wise_squared_loss)
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())
dynamic_step_driver.DynamicStepDriver(
    train_env,
    random_policy,
    observers=[replay_buffer.add_batch],
    num_steps=initial_collect_steps
).run()

dataset = replay_buffer.as_dataset(num_parallel_calls=3,
                                   sample_batch_size=batch_size,
                                   num_steps=2).prefetch(3)
iterator = iter(dataset)

agent.train = common.function(agent.train)

for _ in tqdm(range(num_iterations), desc='Training DQN'):
    dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration).run()
    experience, _ = next(iterator)
    agent.train(experience)

create_policy_eval_video(agent.policy, env_name, num_eval_episodes)
