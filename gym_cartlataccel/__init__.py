from gymnasium.envs.registration import register

register(
  id="CartLatAccel-v0",
  entry_point="gym_cartlataccel.env:BatchedCartLatAccelEnv"
)

register(
  id="CartLatAccel-v1",
  entry_point="gym_cartlataccel.env_v1:CartLatAccelEnv"
)
