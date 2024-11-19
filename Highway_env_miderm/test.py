import gymnasium as gym
import highway_env
import tensorflow as tf
import numpy as np

# 載入訓練好的模型
model = tf.keras.models.load_model("optimized_model.h5")

# 建立環境並設置 render_mode 為 "human" 來顯示可視化
env = gym.make("roundabout-v0", render_mode="human")

# 載入測試數據（假設你有驗證數據可以作為測試數據使用）
test_data = np.load("validation.npz")  # 使用驗證集代替測試集
true_labels = test_data["label"]  # 真實標籤
test_inputs = test_data["data"]  # 測試輸入數據

# 進行模型預測以計算準確率
predictions = model.predict(test_inputs)
predicted_labels = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_labels == true_labels)  # 準確率計算

# 設定測試的回合數與回饋相關變數
num_episodes = 10  # 測試回合數
total_rewards = []  # 用於存放每回合總回饋
reward_matrix = []  # 用於存放回合回饋的整數矩陣

for episode in range(num_episodes):
    obs, _ = env.reset()
    obs_data = np.pad(obs[0].flatten(), (0, 25 - obs[0].size), 'constant').reshape(1, 25)
    done = False
    episode_reward = 0

    while not done:
        # 模型預測動作
        action_probs = model.predict(obs_data, verbose=0)
        action = np.argmax(action_probs)

        # 執行動作並得到新的狀態和回饋
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 無碰撞的情況下增加穩定行駛獎勵
        if not info.get("crashed", False):
            reward = 0.91  # 穩定行駛獎勵
        else:
            reward = -10  # 如果發生碰撞，則施加懲罰

        # 累積回饋
        episode_reward += reward
        obs_data = np.pad(obs[0].flatten(), (0, 25 - obs[0].size), 'constant').reshape(1, 25)

    # 記錄回合結果
    total_rewards.append(episode_reward)
    reward_matrix.append(round(episode_reward))  # 四捨五入存入矩陣
    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

env.close()

# 顯示回饋矩陣
print("Reward Matrix:")
print(reward_matrix)

# 顯示平均回饋
average_reward = np.mean(total_rewards)
print(f"Average Reward over {num_episodes} episodes: {average_reward}")

# 顯示模型準確率
print(f"Validation Accuracy: {accuracy:.4f}")