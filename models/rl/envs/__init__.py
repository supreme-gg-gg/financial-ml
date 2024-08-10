from gymnasium.envs.registration import register

register(id='trading-v0', entry_point='models.rl.gdqn.env:TradingEnv', timestep_limit=1000)