from gymnasium.envs.registration import register

register(id='trading-v0', entry_point='models.rl.envs.env:TradingEnv')
