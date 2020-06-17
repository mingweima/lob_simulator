import logging
import os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sim_env.trading_env import Simulator

def ac_train(hyperparameters, ac_dict, ob_dict, train_days, test_days):
    ticker = hyperparameters['ticker']
    look_back = hyperparameters['lstm_lookback']
    liquidate_volume = hyperparameters['liquidate_volume']
    price_smooth = hyperparameters['price_smooth']

    stocks = {
        '002095': 100000 * liquidate_volume,
    }
    initial_shares = {}
    initial_shares[ticker] = stocks[ticker]

    NUM_OF_STEPS = 12

    t = time.strftime('%Y-%m-%d_%H:%M:%I', time.localtime(time.time()))
    dirpath = os.getcwd() + '/recordings/all/loop{}_{}'.format(hyperparameters['total_loop'], t)

    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    almgren_chriss_f = open(dirpath + '/almgren_chriss.txt', 'w+')

    train_env = {}
    for ticker in initial_shares.keys():
        train_dict = {}
        for day in train_days:
                train_dict[day] = {}
                for session in ['morning', 'afternoon']:
                    with open(os.getcwd() +
                              '/data/{}/{}_{}.txt'.format(ticker, day, session),
                              'rb') as df:
                        data = pickle.load(df, encoding='iso-8859-1')
                        train_dict[day][session] = data
        train_env[ticker] = Simulator(train_dict, ac_dict, ob_dict, initial_shares[ticker], look_back,
                                      price_smooth)
    num_of_training_days = len(train_days)

    test_env = {}
    for ticker in initial_shares.keys():
        test_dict = {}
        for day in test_days:
                test_dict[day] = {}
                for session in ['morning', 'afternoon']:
                    with open(os.getcwd() +
                              '/data/{}/{}_{}.txt'.format(ticker, day, session),
                              'rb') as df:
                        data = pickle.load(df, encoding='iso-8859-1')
                        test_dict[day][session] = data
        test_env[ticker] = Simulator(test_dict, ac_dict, ob_dict, initial_shares[ticker], look_back, price_smooth)
    num_of_test_days = len(train_days)

    for f in [None, almgren_chriss_f]:
        print('Training Set Num of Days: ', num_of_training_days, file=f)
        print('Test Set Num of Days: ', num_of_test_days, file=f)
        print('============================================================', file=f)
        print('Running Almgren Chriss!', file=f)

    def almgren_chriss(kappa, ac_dict, step, num_of_steps):

        if kappa == 0:
            nj = 1 / num_of_steps
        else:
            nj = 2 * np.sinh(0.5 * kappa) * np.cosh(kappa * (
                    num_of_steps - (step - 0.5))) / np.sinh(kappa * num_of_steps)
        # action = closest_action(nj)
        return nj

    kappas = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for kappa in kappas:
        bar = tqdm(range(num_of_training_days * 2 * len(list(initial_shares.keys()))), leave=False)
        bar.set_description('AC Training Set, kappa = {}'.format(kappa))
        print('Training Set, kappa = {}'.format(kappa), file=almgren_chriss_f)
        rewards = []
        for ticker in initial_shares.keys():
            res = []
            ps = []
            acs = []
            dones = []
            for day in train_days:
                    for session in ['morning', 'afternoon']:
                        step = 1
                        bar.update(1)
                        train_env[ticker].reset(day, session)
                        total_reward = 0
                        while True:
                            nj = almgren_chriss(kappa, ac_dict, step, NUM_OF_STEPS)
                            step += 1
                            state, reward, done, info = train_env[ticker].step(nj, ac=True)
                            total_reward += reward
                            res.append(reward)
                            acs.append(info['size'])
                            ps.append(info['price'])
                            if done:
                                dones.append(len(res) - 1)
                                break
                        rewards.append(total_reward)
                        print(ticker, ', {}, {} Total Reward: '.format(day, session), round(total_reward, 3),
                              file=almgren_chriss_f)
            fig = plt.figure(figsize=(40, 20))
            reward_plot = fig.add_subplot(311)
            reward_plot.plot(res)
            reward_plot.set_title('Reward')
            ac_plot = fig.add_subplot(312)
            color = ['b'] * len(acs)
            for indx in dones:
                color[indx] = 'r'
            ac_plot.bar(range(len(acs)), acs, color=color)
            ac_plot.set_title('Action')
            p_plot = fig.add_subplot(313)
            p_plot.set_title('Price')
            p_plot.plot(ps)
            plt.savefig(dirpath + '/AC_train_{}_kappa{}.png'.format(ticker, kappa))
        bar.close()

        for f in [None, almgren_chriss_f]:
            print('Train AC Average, kappa = {}: '.format(kappa), round(np.average(rewards), 3), file=f)
            print('============================================================', file=f)

        print('Test Set, kappa = {}'.format(kappa), file=almgren_chriss_f)
        bar = tqdm(range(num_of_test_days * 2 * len(list(initial_shares.keys()))), leave=False)
        bar.set_description('AC Test Set, kappa = {}'.format(kappa))
        rewards = []
        for ticker in initial_shares.keys():
            res = []
            ps = []
            acs = []
            dones = []
            for day in test_days:
                    for session in ['morning', 'afternoon']:
                        bar.update(1)
                        test_env[ticker].reset(day, session)
                        step = 1
                        total_reward = 0
                        while True:
                            nj = almgren_chriss(kappa, ac_dict, step, NUM_OF_STEPS)
                            step += 1
                            state, reward, done, info = test_env[ticker].step(nj, ac=True)
                            total_reward += reward
                            res.append(reward)
                            acs.append(info['size'])
                            ps.append(info['price'])
                            if done:
                                dones.append(len(res) - 1)
                                break
                        rewards.append(total_reward)
                        print(ticker, ', {}, {} Total Reward: '.format(day, session), round(total_reward, 3),
                              file=almgren_chriss_f)

            fig = plt.figure(figsize=(40, 20))
            reward_plot = fig.add_subplot(311)
            reward_plot.plot(res)
            reward_plot.set_title('Reward')
            ac_plot = fig.add_subplot(312)
            color = ['b'] * len(acs)
            for indx in dones:
                color[indx] = 'r'
            ac_plot.bar(range(len(acs)), acs, color=color)
            ac_plot.set_title('Action')
            p_plot = fig.add_subplot(313)
            p_plot.set_title('Price')
            p_plot.plot(ps)
            plt.savefig(dirpath + '/AC_test_{}_kappa{}.png'.format(ticker, kappa))

            AC_list_f = open(dirpath + '/{}_ACtest_res_kappa{}.txt'.format(ticker, kappa), 'wb')
            pickle.dump(res, AC_list_f)
            AC_list_f.close()
            AC_list_f = open(dirpath + '/ticker_ACtest_acs_kappa{}.txt'.format(ticker, kappa), 'wb')
            pickle.dump(acs, AC_list_f)
            AC_list_f.close()

        bar.close()

        for f in [None, almgren_chriss_f]:
            print('Test AC Average, kappa = {}: '.format(kappa), round(np.average(rewards), 3), file=f)
            print('============================================================', file=f)

