import pickle
from tqdm import tqdm
import argparse
import pandas as pd

# parser = argparse.ArgumentParser()
# parser.add_argument('--ticker', type=str, default='002095')
# args = parser.parse_args()

ticker = '002095'
gap_in_seconds = 300
skip_rows = 10
data_path = '/Users/mmw/Documents/GitHub/lob_simulator/data'


days = [f'2016030{x}' for x in range(10)]

for day in days:

    bar = tqdm(range(7))
    bar.set_description('Reading Data -- {}'.format(day))
    path_name = '{}/{}/{}_{}.csv'.format(data_path, ticker, ticker, day)

    try:
        raw_data = pd.read_csv(path_name, encoding='gbk', error_bad_lines=False)
    except FileNotFoundError:
        continue

    bar.update(1)
    bar.set_description('Dropping Columns -- {}'.format(day))

    data = raw_data.drop(['代码',
                          '现价','当笔成交量','当笔成交金额','成交笔数','IOPV(基金)利息(债券)','成交标志',
                          'BS标志','当日累计成交量','当日成交额（元）','加权平均叫卖价（上海L2）','加权平均叫买价（上海L2）','叫卖总量（上海L2）','叫买总量（上海L2）'], axis=1)
    data = data.iloc[skip_rows:]

    bar.update(1)
    bar.set_description('Rounding to Time Integer -- {}'.format(day))
    data['Date-Time'] = data['日期'] + 'T' + data['时间']
    data['Date-Time'] = pd.to_datetime(data['Date-Time'],
                                       format='%Y-%m-%dT%H:%M:%S').dt.round('{}s'.format(gap_in_seconds))
    data = data.drop(['日期', '时间'], axis=1)
    bar.update(1)
    bar.set_description('Grouping By -- {}'.format(day))

    data = data.groupby(['Date-Time']).first().reset_index()

    bar.update(1)
    # bar.set_description('Deleting Weekends -- {}'.format(day))

    data['Day'] = data['Date-Time'].dt.dayofweek
    # data = data.drop(data.loc[(data['Day'] == 5) | (data['Day'] == 6)].index)
    # data = data.drop(data.loc[(data['Date-Time'] >= pd.to_datetime('2016/7/14'))
    #                           & (data['Date-Time'] < pd.to_datetime('2016/7/15'))].index)
    # data = data.drop(data.loc[(data['Date-Time'] >= pd.to_datetime('2018/4/5'))
    #                           & (data['Date-Time'] < pd.to_datetime('2018/4/7'))].index)
    # for year in [2016, 2017, 2018]:
    #     data = data.drop(data.loc[(data['Date-Time'] >= pd.to_datetime('{}/12/23'.format(year)))
    #                        & (data['Date-Time'] < pd.to_datetime('{}/12/29'.format(year)))].index)

    bar.update(1)
    bar.set_description('Deleting Auction Periods -- {}'.format(day))

    data['Hour'] = data['Date-Time'].dt.hour
    data['Minute'] = data['Date-Time'].dt.minute
    data = data.drop(
        data.loc[(data['Hour'] < 10) | (data['Hour'] > 14)].index)
    data = data.drop(['Minute', 'Day'], axis=1)

    bar.update(1)
    bar.set_description('Storing Data -- {}'.format(day))

    date = pd.to_datetime(data['Date-Time'].dt.strftime('%Y/%m/%d'))
    unique_date = pd.unique(date)
    print(unique_date)
    for aday in unique_date:
        for session in ['morning', 'afternoon']:
            df_train = open('{}/{}/{}_{}.txt'.format(data_path, ticker, day, session), 'wb')
            if session == 'morning':
                session_data = data.loc[data['Date-Time'] >= aday + pd.Timedelta('{}hours'.format(8))]
                session_data.reset_index(drop=True, inplace=True)
                # session_data = session_data.iloc[:49]
                # ext = session_data.loc[48]
                # if ext['Hour'] != 12:
                #     print('Day: ', day)
                #     print(session_data)
            else:
                session_data = data.loc[data['Date-Time'] >= aday + pd.Timedelta('{}hours'.format(11))]
                session_data.reset_index(drop=True, inplace=True)
                # session_data = session_data.iloc[:48]
                # ext = session_data.loc[48]
                # if ext['Hour'] != 15:
                #     print('Day: ', aday)
                #     print(session_data)

            pickle.dump(session_data, df_train)
            df_train.close()

    bar.update(1)
    bar.set_description('Finished Processing Data -- {}'.format(day))
    bar.close()