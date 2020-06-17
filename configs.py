import argparse

import matplotlib.pyplot as plt

from almgren_chriss.almgren_chriss_train import ac_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='ac')
    parser.add_argument('--total_loop', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--explore_stop', type=float, default=0.05)
    parser.add_argument('--decay_rate', type=float, default=0.05)
    parser.add_argument('--target_network_update', type=int, default=500)
    parser.add_argument('--memory_size', type=int, default=100000)
    parser.add_argument('--network_update', type=int, default=1)
    parser.add_argument('--ticker', type=str, default='002095')
    parser.add_argument('--lstm_lookback', type=int, default=12)
    parser.add_argument('--liquidate_volume', type=float, default=0.05)
    parser.add_argument('--num_of_train_months', type=int, default=5)
    parser.add_argument('--num_of_test_months', type=int, default=2)
    parser.add_argument('--price_smooth', type=int, default=10)

    args = parser.parse_args()
    hyperparameters = {
        'ticker': args.ticker,
        'total_loop': args.total_loop,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'explore_stop': args.explore_stop,
        'decay_rate': args.decay_rate,
        'target_network_update': args.target_network_update,
        'memory_size': args.memory_size,
        'network_update': args.network_update,
        'lstm_lookback': args.lstm_lookback,
        'liquidate_volume': args.liquidate_volume,
        'num_of_train_months': args.num_of_train_months,
        'num_of_test_months': args.num_of_test_months,
        'price_smooth': args.price_smooth
    }

    train_days = [f'2016030{x}' for x in [1]]

    test_days = [f'2016030{x}' for x in [7]]

    ac_dict = {0: 1, 1: 1.1, 2: 1.2, 3: 1.3, 4: 1.4, 5: 1.5, 6: 1.6, 7: 1.7, 8: 1.8, 9: 1.9, 10: 2.0}

    ob_dict = {
        'Elapsed Time': True,
        'Remaining Inventory': True,
        'Bid Ask Spread 1': True,
        'Bid Ask Spread 2': True,
        'Bid Ask Spread 3': True,
        'Bid Ask Spread 4': True,
        'Bid Ask Spread 5': True,
        'Bid Ask Spread 6': True,
        'Bid Ask Spread 7': True,
        'Bid Ask Spread 8': True,
        'Bid Ask Spread 9': True,
        'Bid Ask Spread 10': True,
        'Bid Price 1': True,
        'Bid Price 2': True,
        'Bid Price 3': True,
        'Bid Price 4': True,
        'Bid Price 5': True,
        'Bid Price 6': True,
        'Bid Price 7': True,
        'Bid Price 8': True,
        'Bid Price 9': True,
        'Bid Price 10': True,
        'Bid Volume 1': True,
        'Bid Volume 2': True,
        'Bid Volume 3': True,
        'Bid Volume 4': True,
        'Bid Volume 5': True,
        'Bid Volume 6': True,
        'Bid Volume 7': True,
        'Bid Volume 8': True,
        'Bid Volume 9': True,
        'Bid Volume 10': True,
        'Ask Price 1': True,
        'Ask Price 2': True,
        'Ask Price 3': True,
        'Ask Price 4': True,
        'Ask Price 5': True,
        'Ask Price 6': True,
        'Ask Price 7': True,
        'Ask Price 8': True,
        'Ask Price 9': True,
        'Ask Price 10': True,
        'Ask Volume 1': True,
        'Ask Volume 2': True,
        'Ask Volume 3': True,
        'Ask Volume 4': True,
        'Ask Volume 5': True,
        'Ask Volume 6': True,
        'Ask Volume 7': True,
        'Ask Volume 8': True,
        'Ask Volume 9': True,
        'Ask Volume 10': True,
    }


    print("============================================================")
    print("Reinforcement Learning for Optimal Execution")
    print("============================================================")
    print("Ticker:                          ", args.ticker)
    print("Volume to Liquidate (%):         ", args.liquidate_volume)
    print("Agent:                           ", args.agent)
    print("Total Loop:                      ", args.total_loop)
    print("Batch Size:                      ", args.batch_size)
    print("Initial Learning Rate:           ", args.learning_rate)
    print("Final Exploration Probability:   ", args.explore_stop)
    print("Exploration Decay:               ", args.decay_rate)
    print("Target Network Update (step):    ", args.target_network_update)
    print("Replay Buffer Size:              ", args.memory_size)
    print("Network Update Period (step):    ", args.network_update)
    print("LSTM Lookback:                   ", args.lstm_lookback)
    print("Number of Train Months:          ", args.num_of_train_months)
    print("Number of Test Months:           ", args.num_of_test_months)
    print("Price Smooth:                    ", args.price_smooth)
    print("============================================================")
    print("Observation Space:               ", ob_dict)
    print("Action Space:                    ", ac_dict)
    print("============================================================")

    if args.agent == 'ac' or args.agent == 'AC':
        ac_train(hyperparameters, ac_dict, ob_dict, train_days, test_days)

