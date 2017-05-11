from matplotlib import style, pylab
import pandas as pd
#import pandas_datareader.data as web
import datetime
import bs4 as bs
import pickle
import requests
import datetime as dt
import os
import sys

pylab.rcParams['figure.figsize'] = (15, 10)
style.use('ggplot')

#
# Dataset manages the data representation, management, and modification.
#
#

class Dataset:

    def __init__(self):
        self.name = None
        self.stock = None
        self.data = None
        self.target = None

    #
    # Add features to the data. Features are used by the machine learning estimators.
    #
    def add_features(self, df):
        df['delta change'] = (df['Adj Close'] - df['Open'])
        df['delta MaxMin'] = (df['High'] - df['Low'])

        for x in [5, 10, 15, 30, 60, 120, 240]:
            df['{}ma'.format(x)] = df["Adj Close"].rolling(window=x).mean()
            df['delta change mean {}'.format(x)] = df['delta change'].rolling(x).mean()
            df['delta MaxMin mean {}'.format(x)] = df['delta MaxMin'].rolling(x).mean()

        df['std delta change'] = (((df['Adj Close'] - df['Open']) - (df['Adj Close'] - df['Open']).mean()) ** 2)
        df['std delta MaxMin'] = (((df['High'] - df['Low']) - (df['High'] - df['Low']).mean()) ** 2)

        #
        # Drop tables
        #
        del (df["Volume"])
        df.dropna(inplace=True)

        #
        # Adds history to data frame
        # percentage change
        #
        hm_days = 5
        for i in range(1, hm_days + 1):
            for feature in ["Adj Close", "std delta change", "std delta MaxMin"]:
                df["hist_{}_{}".format(feature, i)] = (df[feature].shift(i) / df[feature])

        df = (df - df.mean()) / (df.max() - df.min())

        #
        # Target value - what to predict
        # Classes:
        #   target > +1% = Buy
        #   target < +1% = Sell
        # feature:
        #   target feature to predict
        # gain:
        #   How much +% required to buy
        # future_days:
        #   How many days to look ahead
        #
        gain = 0.02
        future_days = 14
        feature = "5ma"
        df["target"] = 1 - (df[feature] / df[feature].shift(-future_days))

        # 1 or 0 ; classification problem
        df["target"] = [round(x + .5 + gain) for x in df["target"]]

        for id, val in enumerate(df["target"]):
            if round(val + .5 + gain) > 1:
                df["target"][id] = 1
            else:
                df["target"][id] = 0

        df.dropna(0, inplace=True)

        self.target = df["target"].tolist()
        del(df["target"])

        return df

    #
    # Get individual stock; called by 'get_stocks'
    #
    def download_data(self, stock, reload=False):
        today = [int(x) for x in str(datetime.date.today()).split("-")]
        start = dt.datetime(2000, 1, 1)
        stop = dt.datetime(today[0], today[1], today[2])

        if not os.path.exists("data/stocks/{}.csv".format(stock)) or reload:
            try:
                df = web.DataReader(stock, 'yahoo', start, stop)
                df.to_csv("data/stocks/{}.csv".format(stock))
                print(stock)
            except:
                e = sys.exc_info()[0]
                print("err", e)
        else:
            print("Already have: {}".format(stock))

    #
    # Read stock data to memory
    #
    def read_stock_data(self):
        if os.path.exists("data/stocks/{}.csv".format(self.stock)):
            df = pd.read_csv('data/stocks/{}.csv'.format(self.stock), parse_dates=True, index_col=0)
            df = self.add_features(df)
            self.data = df

            # plt.plot(df[["Open"]][-365:], label="Open") #, marker='o', markeredgecolor='black', markersize=3)
            # plt.legend(loc="upper left", ncol=2)
            # plt.title(stock_name)
            # plt.savefig('data/plot/{}.png'.format(stock))


#
# Get a list of OMXs30 companies by parsing the Wikipedia list
#
def get_companies():
    resp = requests.get("https://en.wikipedia.org/wiki/OMX_Stockholm_30")
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class': 'wikitable sortable'})
    stocks = []

    for row in table.findAll('tr')[1:]:
        stock = row.findAll('td')[2].text
        stock_name = row.findAll('td')[0].text
        stocks.append((stock, stock_name))

    with open("data/omxs30.pickle", "wb") as f:
        pickle.dump(stocks, f)

    return stocks


#
# Get stock data from the Yahoo API
# self.stocks contain the names of the stocks
#
def get_stocks():
    if not os.path.exists("data/omxs30.pickle"):
        stocks = get_companies()
    else:
        with open("data/omxs30.pickle", "rb") as f:
            stocks = pickle.load(f)

    if not os.path.exists("data/stocks"):
        os.makedirs("data/stocks")

    stock_list = []
    for stock, name in stocks:
        ds = Dataset()
        ds.name = name
        ds.stock = stock
        ds.download_data(stock)
        ds.read_stock_data()

        if ds.data is not None:
            stock_list.append(ds)
        else:
            print("\terr", name, stock)

    return stock_list


# tmp = get_stocks()
#
# print(tmp[0].name)
# print(tmp[0].stock)
# print(tmp[0].data[:3])
# print("---")
# print(tmp[0].target[:3])
