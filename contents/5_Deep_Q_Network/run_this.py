from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0

    for episode in range(10):
        print '=' * 100

        observation = env.reset()

        while True:
            env.render()

            # choose action
            action = RL.choose_action(observation)

            # get reward & get next action
            observation_, reward, done = env.step(action)
            
            # prepare SARS for DNN
            RL.store_transition(observation, action, reward, observation_)

            # reinforcement learning
            if (step > 5) and (step % 5 == 0):
                RL.learn()

            # take action
            observation = observation_

            if done: break
            step += 1

    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()

    RL  = DeepQNetwork(env.n_actions,
                       env.n_features,

                       learning_rate       = 0.01,
                       reward_decay        = 0.9,
                       e_greedy            = 0.9,
                       
                       replace_target_iter = 200,
                       memory_size         = 2000)
                       # output_graph=True)

    env.after(100, run_maze) # wait for initialize
    env.mainloop()           # start environment
    RL.plot_cost()
