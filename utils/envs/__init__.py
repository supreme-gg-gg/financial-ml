from gymnasium.envs.registration import register

register(id='trading-v0', entry_point='utils.envs.env:TradingEnv')
