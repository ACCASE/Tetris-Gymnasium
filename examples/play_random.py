import gymnasium as gym
import cv2

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)

    terminated = False
    while not terminated:
        key = cv2.waitKey(1)
        # print(env.render() + "\n")
        # print(env.render(), end="\n\n")
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
    print("Game Over!")
