import os.path
from collections import OrderedDict
from datetime import timedelta
import numpy as np
import pandas as pd
from linear_regression import LinearRegression

PATH_GIVEN_TRAIN = './given/train.csv'
PATH_GIVEN_TEST = './given/test_X.csv'
PATH_TRAIN_DATA = './result/train_data.csv'
PATH_VALID_DATA = './result/valid_data.csv'

def _create_tidy_training_set():
    # read raw data from train data and label it
    # 0-23 mean hour of the day
    colnames = ['DATE','SITE','ITEM']+list(map(str,range(24)))
    df = pd.read_csv(PATH_GIVEN_TRAIN, names=colnames, skiprows=1) # skip the first line

    # remove column 'SITE'
    df = df.loc[:,['DATE','ITEM'] + list(map(str,range(24)))]

    # melt 'hour' to column
    df = pd.melt(df,
                 id_vars=['DATE','ITEM'],
                 value_vars=list(map(str,range(24))),
                 var_name='HOUR',
                 value_name='VALUE')

    # snapshot:
    # columns : DATE, ITEM, HOUR, VALUE

    # generate 'DATETIME'

    df['DATETIME'] = pd.to_datetime(df.DATE + ' ' + df.HOUR + ':00:00')
    df = df.loc[:,['DATETIME', 'ITEM', 'VALUE']]

    # snapshot:
    # columns : DATETIME, ITEM, VALUE

    # replace NR to 0
    df.loc[df.VALUE == 'NR', 'VALUE'] = 0

    # change 'VALUE' type
    df['VALUE'] = df['VALUE'].astype(float)

    # pivot 'ITEM' to COLUMNS
    df = df.pivot_table(values='VALUE',
                        index='DATETIME',
                        columns='ITEM',
                        aggfunc='sum')

    # snapshot:
    # index: DATETIME
    # columns : AMB_TEMP, CH4, CO, ..., PM2.5, ...

    return df


def _split_df_into_train_and_valid(df):
    df_valid = df.loc[df.index.month == 12, :]
    df_train = df.loc[df.index.month != 12, :]
    return (df_train,df_valid)


def _convert_to_problem_regression_form(df):
    """
    Convert to the problem regression form

    The problem is that using parameters among continuous 9 hours to predict PM2.5 on next hour

    :param df: 'index' is datetime 'columns' includes different parameters
    :return:
    """
    data = OrderedDict()

    # create columns of data
    param_list = df.columns.tolist()
    for i in range(9):
        for param in param_list:
            data['{:02d}h__{}'.format(i+1,param)] = []

    data['10h__PM2.5']=[]

    # add content into 'data'
    datetime_list = df.index
    d1h = timedelta(hours=1)
    for m in pd.unique(datetime_list.month):
        for timestamp in (df.loc[df.index.month == m, :]).index:
            start, end = timestamp, timestamp + 9 * d1h
            sub_df = df.loc[(start <= df.index) & (df.index <= end), :]
            # shape[0] returns num of rows
            if sub_df.shape[0] == 10:
                for i in range(9):
                    for param in param_list:
                        data['{:02d}h__{}'.format(i+1,param)].append(
                            sub_df.loc[timestamp+i*d1h,param])
                data['10h__PM2.5'].append(sub_df.loc[timestamp+9*d1h, 'PM2.5'])
    return pd.DataFrame(data)


def preprocess_training_set():
    if os.path.isfile(PATH_TRAIN_DATA) and os.path.isfile(PATH_VALID_DATA):
        train_df = pd.read_csv(PATH_TRAIN_DATA)
        valid_df = pd.read_csv(PATH_VALID_DATA)
    else:
        df = _create_tidy_training_set()
        tmp_train_df, tmp_valid_df = _split_df_into_train_and_valid(df)

        # print(tmp_valid_df)
        train_df = _convert_to_problem_regression_form(tmp_train_df)
        train_df.to_csv(PATH_TRAIN_DATA, index=None)
        valid_df = _convert_to_problem_regression_form(tmp_valid_df)
        valid_df.to_csv(PATH_VALID_DATA, index=None)


    train_x = np.array(train_df.loc[:, train_df.columns != '10h__PM2.5'])
    train_y = np.array(train_df.loc[:, '10h__PM2.5'])

    valid_x = np.array(valid_df.loc[:, valid_df.columns != '10h__PM2.5'])
    valid_y = np.array(valid_df.loc[:, '10h__PM2.5'])

    col_name = (train_df.loc[:, valid_df.columns != '10h__PM2.5']).columns

    return (train_x,train_y,valid_x,valid_y,col_name)


def preprocess_testing_set(col_name):
    col_names = ['ID','ITEM'] + list(map(lambda  x: '{:02d}h'.format(x),range(1,10)))
    test_df = pd.read_csv(PATH_GIVEN_TEST,names=col_names,header=None)

    # snapshot:
    # columns: ID, ITEM, 01h, 02h, ..., 09h
    for col in list(map(lambda  x: '{:02d}h'.format(x),range(1,10))):
        test_df.loc[(test_df.ITEM == 'RAINFALL') & (test_df[col] == 'NR'),col] = 0

    # ['ID','ITEM','HOUR','VALUE'] form
    test_df = test_df.pivot_table(index=['ID','ITEM'],aggfunc='sum')
    test_df=test_df.stack()
    test_df=test_df.reset_index()
    test_df.columns = ['ID','ITEM','HOUR','VALUE']

    # snapshot
    # columns: ID, ITEM, HOUR, VALUE

    # combine: 'HOUR' and 'ITEM' to 'COL'
    test_df['COL'] = test_df.HOUR + '__' + test_df.ITEM
    test_df = test_df[['ID','COL','VALUE']]

    # snapshot
    # columns: ID, COL, VALUE

    # pivot 'COL' to columns
    test_df = test_df.pivot_table(values='VALUE', index='ID', columns='COL',aggfunc='sum')

    # snapshot
    # index: ID
    # columns: 01h_AMB_TEMP ....

    # re-order
    test_df['IDNUM'] = test_df.index.str.replace('id_','').astype('int')
    test_df = test_df.sort_values(by = 'IDNUM')
    test_df = test_df.drop('IDNUM',axis=1)

    # snapshot:
    # index: ID
    # columns: ID_Num, 01h__AMB_TEMP, 01h__CH4, 01h__CO,..., 09h__WS_HR
    testX = np.array(test_df[col_name],dtype='float64')
    ids = np.array(test_df.index, dtype='str')
    return (testX,ids)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='hw1')

    parser.add_argument('--method', metavar='METHOD', type=str, nargs='?',
                        help='method of regression (\"pseudo_inverse\" or \"gradient_descent\")', required=True),
    parser.add_argument('--output', metavar='OUTPUT', type=str, nargs='?',
                        help='path of result', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)

    trainX, trainY, validX, validY, colName = preprocess_training_set()

    linreg = LinearRegression()

    if args.method == 'pseudp_inverse':
        linreg.train_by_pseudo_inverse(
            trainX,trainY,alpha=0.5,validate_data=(validX,validY)
        )
    elif args.method == 'gradient_descent':
        linreg.train_by_gradient_descent(
            trainX,trainY,epoch=1000,rate=0.000001,batch=100,alpha=0.00000001,
            validate_data=(validX,validY)
        )
    else:
        raise Exception('wrong method')

    testX, ids = preprocess_testing_set(colName)
    predY = linreg.predict(testX)

    result = list()

    for i in range(ids.shape[0]):
        result.append([ids[i],predY[i]])

    with open(args.output, 'w') as fw:
        for id, pred in result:
            fw.write('{id},{pred}\n'.format(id=id,pred=pred))


if __name__ == '__main__':
    main()