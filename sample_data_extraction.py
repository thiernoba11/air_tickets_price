# system-library
import json
import os
import numpy as np
from datetime import datetime


"""
# data prepare for the dataset1
"""
routes_dataset1 = ["BCN_BUD",  # route 1
                  "BUD_BCN",  # route 2
                  "CRL_OTP",  # route 3
                  "MLH_SKP",  # route 4
                  "MMX_SKP",  # route 5
                  "OTP_CRL",  # route 6
                  "SKP_MLH",  # route 7
                  "SKP_MMX"]  # route 8
# for currency change - change different currency to Euro
currency_dataset1 = [1,      # route 1 - Euro
                     0.0032, # route 2 - Hungarian Forint
                     1,      # route 3 - Euro
                     1,      # route 4 - Euro
                     0.12,   # route 5 - Swedish Krona
                     0.25,   # route 6 - Romanian Leu
                     0.018,  # route 7 - Macedonian Denar
                     0.018   # route 8 - Macedonian Denar
                     ]

"""
# data prepare for the dataset2
"""
routes_dataset2 = ["BGY_OTP", # route 1
                    "BUD_VKO", # route 2
                    "CRL_OTP", # route 3
                    "CRL_WAW", # route 4
                    "LTN_OTP", # route 5
                    "LTN_PRG", # route 6
                    "OTP_BGY", # route 7
                    "OTP_CRL", # route 8
                    "OTP_LTN", # route 9
                    "PRG_LTN", # route 10
                    "VKO_BUD", # route 11
                    "WAW_CRL"] # route 12

# for currency change - change different currency to Euro
currency_dataset2 = [1,      # route 1 - Euro
                     0.0032, # route 2 - Hungarian Forint
                     1,      # route 3 - Euro
                     1,      # route 4 - Euro
                     1,      # route 5 - Euro
                     1,      # route 6 - Euro
                     0.25,   # route 7 - Romanian Leu
                     0.25,    # route 8 - Romanian Leu
                     0.25,   # route 9 - Romanian Leu
                     0.037,  # route 10 - Czech Republic Koruna
                     1,      # route 11 - Euro
                     0.23    # route 12 - Polish Zloty
                     ]

def remove_duplicates(values):
    """
    remove duplicate value in a list
    :param values: input list
    :return: no duplicate entry list
    """
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet,
        # ... add it to both list and set.
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def days_between(d1, d2):
    """
    get the days interval between two dates
    :param d1: date1
    :param d2: date2
    :return: days interval
    """
    d1 = datetime.strptime(d1, "%Y%m%d")
    d2 = datetime.strptime(d2, "%Y%m%d")
    return abs((d2 - d1).days)

def getPrice(price):
    """
    Get the numeric price in a string format, which contains currency symbol
    :param price:
    :return:
    """
    price = float( filter( lambda x: x in '0123456789.', price) )
    return price

def is_not_nullprice(data):
    """
    used by the filter to filter out the null entries
    :param data: input data
    :return: true if it's not null, false if null
    """
    return data and data["MinimumPrice"] != None

def check_if_only_one_flightNum(datas):
    """
    check whether the datas only contain one flight number
    :param datas: input data
    :return: Ture if the datas only contain one flight number, False otherwise
    """
    kinds = []
    for data in datas:
        kinds += data["Flights"]

    flightNums = []
    for kind in kinds:
        flightNums.append(kind["FlightNumber"])

    if len(remove_duplicates(flightNums)) == 1:
        return True
    else:
        return False



def load_data_with_prefix_and_dataset(filePrefix="BCN_BUD", dataset="dataset1"):
    """
    load the data in the 'dataset' with 'filePrefix'
    :param filePrefix: choose which route
    :param dataset: dataset name('dataset1' or 'dataset2')
    :return: decoded data
    """
    currentDir = os.path.dirname(os.path.realpath(__file__))
    observeDatesDirs = os.listdir(currentDir + "/data_air/" + dataset) # path directory of each observed date in the dataset

    filePaths = [] # keep all the file paths start with "filePrefix"
    data_decoded = [] # keep all the schedules start with "filePrefix"

    for date in observeDatesDirs:
        currentPath = currentDir + "/data_air/" + dataset + "/" + date

        try:
            files = os.listdir(currentPath) # file names in currect date directory
            for file in files:
                try:
                    if filePrefix in file:
                        filePath = os.path.join(currentPath, file)
                        filePaths.append(filePath)

                        fp = open(filePath, 'r')
                        datas_with_specific_date = json.load(fp)
                        # add observed data
                        for data in datas_with_specific_date:
                            #"Date" is the departure date, "ObservedDate" is the observed date
                            data["ObservedDate"] = date.replace("-", "")
                            data["State"] = days_between(data["Date"], data["ObservedDate"]) - 1
                        data_decoded += datas_with_specific_date # do not use append function

                except:
                    print "Not a json file"
        except:
            print "Not a directory, MAC OS contains .DS_Store file."

    # filter the null entries
    data_decoded = filter(is_not_nullprice, data_decoded)

    return data_decoded


def load_data_with_daysBeforeTakeoff_and_sameFlightNum(days, filePrefix="BCN_BUD", dataset="dataset1"):
    """
    Load data with same flight number and the same days before takeoff.
    i.e. same equivalence class
    But in out dataset, one route means one flight number.
    :param days: the days before takeoff
    :param filePrefix: choose which route
    :param dataset: dataset name('dataset1' or 'dataset2')
    :return: data with same flight number and the same days before takeoff
    """
    datas = load_data_with_prefix_and_dataset(filePrefix, dataset)
    output = [data for data in datas if days_between(data["ObservedDate"], data["Date"]) == days]

    return output


def getMinimumPrice(datas):
    """
    Given the dataset, return the minimum price in the dataset
    :param datas: input dataset
    :return: minimum price in the dataset
    """
    minimumPrice = getPrice(datas[0]["MinimumPrice"]) # in our json data files, MinimumPrice means the price in that day
    for data in datas:
        price = getPrice(data["MinimumPrice"])
        minimumPrice = price if price<minimumPrice else minimumPrice
    minimumPrice = minimumPrice

    return minimumPrice


def getMaximumPrice(datas):
    """
    Given the dataset, return the maximum price in the dataset
    :param datas: input dataset(in QLearning and Neural Nets, it should have same departure date)
    :return: maximum price in the dataset
    """
    maximumPrice = getPrice(datas[0]["MinimumPrice"]) # in our json data files, MinimumPrice means the price in that day
    for data in datas:
        price = getPrice(data["MinimumPrice"])
        maximumPrice = price if price>maximumPrice else maximumPrice

    return maximumPrice

def getChosenPrice(state, datas):
    """
    Given the state, i.e. the days before departure, and the dataset, return the price
    :param state: the days before departure
    :param datas: input dataset(in QLearning, it should have same departure date)
    :return: the chosen price
    """
    for data in datas:
        if data["State"] == state:
            return getPrice(data["MinimumPrice"])

def getMinimumPreviousPrice(departureDate, state, datas):
    """
    Get the minimum previous price, corresponding to the departure date and the observed date
    :param departureDate: departure date
    :param state: observed date
    :param datas: datasets
    :return: minimum previous price
    """
    specificDatas = []
    specificDatas = [data for data in datas if data["Date"]==departureDate]

    minimumPreviousPrice = getPrice(specificDatas[0]["MinimumPrice"])
    for data in specificDatas:
        if getPrice(data["MinimumPrice"]) < minimumPreviousPrice and data["State"]>=state:
            minimumPreviousPrice = getPrice(data["MinimumPrice"])

    return minimumPreviousPrice

def getMaximumPreviousPrice(departureDate, state, datas):
    """
    Get the maximum previous price, corresponding to the departure date and the observed date
    :param departureDate: departure date
    :param state: observed date
    :param datas: datasets
    :return: maximum previous price
    """
    specificDatas = []
    specificDatas = [data for data in datas if data["Date"]==departureDate]

    maximumPreviousPrice = getPrice(specificDatas[0]["MinimumPrice"])
    for data in specificDatas:
        if getPrice(data["MinimumPrice"]) > maximumPreviousPrice and data["State"]>=state:
            maximumPreviousPrice = getPrice(data["MinimumPrice"])

    return maximumPreviousPrice

"""
# step 1. The main data load function - for classification for specific dataset
"""
def load_for_classification_for_Specific(dataset="dataset1", routes=routes_dataset1):
    """
    Load the data for classification
    :param dataset: dataset name('dataset1' or 'dataset2')
    :return: X_train, y_train, X_test, y_test
    """
    isOneOptimalState = False
    # Construct the input data
    dim = routes.__len__() + 4
    X_train = np.empty(shape=(0, dim))
    y_train = np.empty(shape=(0,1))
    y_train_price = np.empty(shape=(0,1))
    X_test = np.empty(shape=(0,dim))
    y_test = np.empty(shape=(0,1))
    y_test_price = np.empty(shape=(0,1))

    for filePrefix in routes:
        datas = load_data_with_prefix_and_dataset(filePrefix, dataset)
        for data in datas:
            print "Construct route {}, State {}, departureDate {}...".format(filePrefix, data["State"], data["Date"])
            x_i = []
            # feature 1: flight number -> dummy variables
            for i in range(len(routes)):
                """
                !!!need to change!
                """
                if i == routes.index(filePrefix):
                    x_i.append(1)
                else:
                    x_i.append(0)

            # feature 2: departure date interval from "20151109", because the first observed date is 20151109
            departureDate = data["Date"]
            """
            !!!maybe need to change the first observed date
            """
            departureDateGap = days_between(departureDate, "20151109")
            x_i.append(departureDateGap)

            # feature 3: observed days before departure date
            state = data["State"]
            x_i.append(state)

            # feature 4: minimum price before the observed date
            minimumPreviousPrice = getMinimumPreviousPrice(data["Date"], state, datas)
            x_i.append(minimumPreviousPrice)

            # feature 5: maximum price before the observed date
            maximumPreviousPrice = getMaximumPreviousPrice(data["Date"], state, datas)
            x_i.append(maximumPreviousPrice)

            # output
            y_i = [0]
            specificDatas = [data2 for data2 in datas if data2["Date"]==departureDate]


            # multiple entries can be buy
            minPrice = getMinimumPrice(specificDatas)
            if getPrice(data["MinimumPrice"]) == minPrice:
                y_i = [1]


            # keep price info
            y_price = [getPrice(data["MinimumPrice"])]

            if int(departureDate) < 20160229 and int(departureDate) >= 20151129: # choose date between "20151129-20160229(20160115)" as training data
                X_train = np.concatenate((X_train, [x_i]), axis=0)
                y_train = np.concatenate((y_train, [y_i]), axis=0)
                y_train_price = np.concatenate((y_train_price, [y_price]), axis=0)
            elif int(departureDate) < 20160508 and int(departureDate) >= 20160229: # choose date before "20160508(20160220)" as test data
                X_test = np.concatenate((X_test, [x_i]), axis=0)
                y_test = np.concatenate((y_test, [y_i]), axis=0)
                y_test_price = np.concatenate((y_test_price, [y_price]), axis=0)
            else:
                pass


    """
    remove duplicate rows for train
    """
    tmp_train = np.concatenate((X_train, y_train, y_train_price), axis=1)
    new_array = [tuple(row) for row in tmp_train]
    tmp_train = np.unique(new_array)

    # get the result
    X_train = tmp_train[:, 0:12]
    y_train = tmp_train[:, 12]
    y_train_price = tmp_train[:, 13]

    """
    remove duplicate rows for test
    """
    tmp_test = np.concatenate((X_test, y_test, y_test_price), axis=1)
    new_array = [tuple(row) for row in tmp_test]
    tmp_test = np.unique(new_array)

    # get the result
    X_test = tmp_test[:, 0:12]
    y_test = tmp_test[:, 12]
    y_test_price = tmp_test[:, 13]

    # save the result
    np.save('./Classification/raw/X_train', X_train)
    np.save('./Classification/raw/y_train', y_train)
    np.save('./Classification/raw/y_train_price', y_train_price)
    np.save('./Classification/raw/X_test', X_test)
    np.save('./Classification/raw/y_test', y_test)
    np.save('./Classification/raw/y_test_price', y_test_price)

    return X_train, y_train, X_test, y_test




"""
# step 2. price normalize for the classification input
"""
def priceNormalize(routes=routes_dataset1, currency=currency_dataset1):
    """
    Different routes have different units for the price, normalize it as Euro. see 'currency_dataset1' for the currency exchange.
    :return: NA
    example: priceNormalize
    """
    """
    Get the input specific clf data for the training data set
    """
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    X_train = np.load('./Classification/raw/X_train.npy')
    y_train = np.load('./Classification/raw/y_train.npy')
    y_train_price = np.load('./Classification/raw/y_train_price.npy')

    # normalize feature 10, feature 11, feature 13
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: prediction(buy or wait); feature 13: price
    evalMatrix_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    matrixTrain = np.empty(shape=(0, evalMatrix_train.shape[1]))
    for i in range(len(routes)):
        evalMatrix = evalMatrix_train[np.where(evalMatrix_train[:, i]==1)[0], :]
        evalMatrix[:, 10] *= currency[i]
        evalMatrix[:, 11] *= currency[i]
        evalMatrix[:, 13] *= currency[i]
        matrixTrain = np.concatenate((matrixTrain, evalMatrix), axis=0)

    X_train = matrixTrain[:, 0:12]
    y_train = matrixTrain[:, 12]
    y_train_price = matrixTrain[:, 13]

    y_train = y_train.reshape((y_train.shape[0], 1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0], 1))


    np.save('./Classification/pricenormalized/X_train', X_train)
    np.save('./Classification/pricenormalized/y_train', y_train)
    np.save('./Classification/pricenormalized/y_train_price', y_train_price)

    """
    Get the input specific clf data for the test data set
    """
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    X_test = np.load('./Classification/raw/X_test.npy')
    y_test = np.load('./Classification/raw/y_test.npy')
    y_test_price = np.load('./Classification/raw/y_test_price.npy')

    # normalize feature 10, feature 11, feature 13
    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: prediction(buy or wait); feature 13: price
    evalMatrix_test = np.concatenate((X_test, y_test, y_test_price), axis=1)
    evalMatrix_test = evalMatrix_test[np.where(evalMatrix_test[:,8]>=20)[0], :]

    matrixTest = np.empty(shape=(0, evalMatrix_test.shape[1]))
    for i in range(len(routes)):
        evalMatrix = evalMatrix_test[np.where(evalMatrix_test[:, i]==1)[0], :]
        evalMatrix[:, 10] *= currency[i]
        evalMatrix[:, 11] *= currency[i]
        evalMatrix[:, 13] *= currency[i]
        matrixTest = np.concatenate((matrixTest, evalMatrix), axis=0)

    X_test = matrixTest[:, 0:12]
    y_test = matrixTest[:, 12]
    y_test_price = matrixTest[:, 13]

    y_test = y_test.reshape((y_test.shape[0], 1))
    y_test_price = y_test_price.reshape((y_test_price.shape[0], 1))


    np.save('./Classification/pricenormalized/X_test', X_test)
    np.save('./Classification/pricenormalized/y_test', y_test)
    np.save('./Classification/pricenormalized/y_test_price', y_test_price)



"""
# step 3. get the regression input and output from classification inputs
"""
def getRegressionOutput_train(routes=routes_dataset1):
    """
    Get the regression output formula from the classification datasets.
    :return: Save the regression datasets into inputGeneralReg
    """
    X_train = np.load('./Classification/pricenormalized/X_train.npy')
    y_train = np.load('./Classification/pricenormalized/y_train.npy')
    y_train_price = np.load('./Classification/pricenormalized/y_train_price.npy')

    # concatenate the buy or wait info to get the total datas
    y_train = y_train.reshape((y_train.shape[0],1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0],1))

    # feature 0~7: flight numbers
    # feature 8: departure date;  feature 9: observed date state
    # feature 10: minimum price; feature 11: maximum price
    # feature 12: prediction(buy or wait); feature 13: current price
    X_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    """
    # define the variables needed to be changed
    """
    dim = 14
    idx_departureDate = 8
    idx_minimumPrice = 10
    idx_output = 12
    idx_currentPrice = 13

    # Construct train data
    X_tmp = np.empty(shape=(0, dim))
    for flightNum in range(len(routes)):

        # choose one route datas
        X_flightNum = X_train[np.where(X_train[:, flightNum]==1)[0], :]

        # group by the feature: departure date
        departureDates_train = np.unique(X_flightNum[:, idx_departureDate])

        # get the final datas, the observed data state should be from large to small(i.e. for time series)
        for departureDate in departureDates_train:
            indexs = np.where(X_flightNum[:, idx_departureDate]==departureDate)[0]
            datas = X_flightNum[indexs, :]
            minPrice = min(datas[:, idx_minimumPrice]) # get the minimum price for the output
            datas[:, idx_output] = minPrice
            """
            print departureDate
            print minPrice
            print datas
            """
            X_tmp = np.concatenate((X_tmp, datas), axis=0)

    X_train = X_tmp[:, 0:idx_output]
    y_train = X_tmp[:, idx_output]
    y_train_price = X_tmp[:, idx_currentPrice]
    y_train = y_train.reshape((y_train.shape[0], 1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0], 1))

    # regression has one more feature than classification
    X_train = np.concatenate((X_train, y_train_price), axis=1)
    np.save('./Regression/X_train', X_train)
    np.save('./Regression/y_train', y_train)
    np.save('./Regression/y_train_price', y_train_price)

def getRegressionOutput_test(routes=routes_dataset1):
    """
    Get the regression output formula from the classification datasets.
    :return: Save the regression datasets into inputGeneralReg
    """
    X_test = np.load('./Classification/pricenormalized/X_test.npy')
    y_test = np.load('./Classification/pricenormalized/y_test.npy')
    y_test_price = np.load('./Classification/pricenormalized/y_test_price.npy')

    # concatenate the buy or wait info to get the total datas
    y_test = y_test.reshape((y_test.shape[0],1))
    y_test_price = y_test_price.reshape((y_test_price.shape[0],1))

    # feature 0~7: flight numbers
    # feature 8: departure date;  feature 9: observed date state
    # feature 10: minimum price; feature 11: maximum price
    # feature 12: prediction(buy or wait); feature 13: current price
    X_test = np.concatenate((X_test, y_test, y_test_price), axis=1)

    """
    # define the variables needed to be changed
    """
    dim = 14
    idx_departureDate = 8
    idx_minimumPrice = 10
    idx_output = 12
    idx_currentPrice = 13

    # Construct train data
    X_tmp = np.empty(shape=(0, dim))
    for flightNum in range(len(routes)):

        # choose one route datas
        X_flightNum = X_test[np.where(X_test[:, flightNum]==1)[0], :]

        # group by the feature: departure date
        departureDates_test = np.unique(X_flightNum[:, idx_departureDate])

        # get the final datas, the observed data state should be from large to small(i.e. for time series)
        for departureDate in departureDates_test:
            indexs = np.where(X_flightNum[:, idx_departureDate]==departureDate)[0]
            datas = X_flightNum[indexs, :]
            minPrice = min(datas[:, idx_minimumPrice]) # get the minimum price for the output
            datas[:, idx_output] = minPrice
            """
            print departureDate
            print minPrice
            print datas
            """
            X_tmp = np.concatenate((X_tmp, datas), axis=0)

    X_test = X_tmp[:, 0:idx_output]
    y_test = X_tmp[:, idx_output]
    y_test_price = X_tmp[:, idx_currentPrice]
    y_test = y_test.reshape((y_test.shape[0], 1))
    y_test_price = y_test_price.reshape((y_test_price.shape[0], 1))

    # regression has one more feature than classification
    X_test = np.concatenate((X_test, y_test_price), axis=1)
    np.save('./Regression/X_test', X_test)
    np.save('./Regression/y_test', y_test)
    np.save('./Regression/y_test_price', y_test_price)




"""
# step 4. visualize for classification
"""
def visualizeData_for_SpecificClassification(filePrefix, isTrain=True, routes=routes_dataset1):
    """
    Visualize the train buy entries for every departure date, for each route
    :param filePrefix: route prefix
    :return: NA
    example: visualizeData_for_SpecificClassification(routes_specific[1], routes_specific)
    """
    if isTrain:
        X_train = np.load('./Classification/pricenormalized/X_train.npy')
        y_train = np.load('./Classification/pricenormalized/y_train.npy')
        y_train_price = np.load('./Classification/pricenormalized/y_train_price.npy')
    else:
        X_train = np.load('./Classification/pricenormalized/X_test.npy')
        y_train = np.load('./Classification/pricenormalized/y_test.npy')
        y_train_price = np.load('./Classification/pricenormalized/y_test_price.npy')

    # route index
    flightNum = routes.index(filePrefix)

    # concatenate the buy or wait info to get the total datas
    y_train = y_train.reshape((y_train.shape[0],1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0],1))

    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: prediction(buy or wait); feature 13: price
    X_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    # choose one route datas
    X_train = X_train[np.where(X_train[:, flightNum]==1)[0], :]

    # remove dummy variables
    # feature 0: departure date;  feature 1: observed date state
    # feature 2: minimum price; feature 3: maximum price
    # feature 4: prediction(buy or wait); feature 5:price
    X_train = X_train[:, 8:14]

    # group by the feature: departure date
    departureDates_train = np.unique(X_train[:, 0])

    # get the final datas, the observed data state should be from large to small(i.e. for time series)
    length_test = []
    for departureDate in departureDates_train:
        indexs = np.where(X_train[:, 0]==departureDate)[0]
        datas = X_train[indexs, :]
        length_test.append(len(datas))
        print departureDate
        print datas



"""
# step 5. visualize for regression - for specific
"""
def visualizeTrainData_for_SpecificRegression(filePrefix, routes):
    """
    Visualize the train buy entries for every departure date, for each route
    :param filePrefix: route prefix
    :return: NA
    example: visualizeTrainData_for_SpecificRegression(routes_general[1], routes_general)
    """
    X_train = np.load('./Regression/X_train.npy')
    y_train = np.load('./Regression/y_train.npy')
    y_train_price = np.load('./Regression/y_train_price.npy')

    X_train2 = np.load('./Regression/X_test.npy')
    y_train2 = np.load('./Regression/y_test.npy')
    y_train2_price = np.load('./Regression/y_test_price.npy')

    X_train = np.concatenate((X_train, X_train2), axis=0)
    y_train = np.concatenate((y_train, y_train2), axis=0)
    y_train_price = np.concatenate((y_train_price, y_train2_price), axis=0)

    """
    define the variables to be changed
    """
    dim = 15
    idx_departureDate = 8


    # route index
    flightNum = routes.index(filePrefix)

    # concatenate the buy or wait info to get the total datas
    y_train = y_train.reshape((y_train.shape[0],1))
    y_train_price = y_train_price.reshape((y_train_price.shape[0],1))

    # feature 0~7: flight number dummy variables
    # feature 8: departure date; feature 9: observed date state;
    # feature 10: minimum price; feature 11: maximum price
    # fearure 12: current price;
    # feature 13: minimum price; feature 14: current price
    X_train = np.concatenate((X_train, y_train, y_train_price), axis=1)

    # choose one route datas
    X_train = X_train[np.where(X_train[:, flightNum]==1)[0], :]

    # remove dummy variables
    # feature 0: departure date;  feature 1: observed date state
    # feature 2: minimum price by now; feature 3: maximum price by now
    # feature 4: current price;
    # feature 5: minimum price; feature 6: current price
    X_train = X_train[:, idx_departureDate:dim]

    # group by the feature: departure date
    departureDates_train = np.unique(X_train[:, 0])

    # get the final datas, the observed data state should be from large to small(i.e. for time series)
    length_test = []
    for departureDate in departureDates_train:
        indexs = np.where(X_train[:, 0]==departureDate)[0]
        datas = X_train[indexs, :]
        length_test.append(len(datas))
        print departureDate
        print datas



if __name__ == "__main__":
    """
    STEP 1: load raw data
    """
    load_for_classification_for_Specific()

    """
    STEP 2: get the data for the classification problem
    """
    priceNormalize()

    """
    STEP 3: get the data for the regression problem
    """
    getRegressionOutput_train()
    getRegressionOutput_test()

    """
    STEP 4: visualize the data set for classification problem
    """
    isTrain = 0
    visualizeData_for_SpecificClassification(routes_dataset1[1], isTrain, routes_dataset1)


    """
    STEP 5: visualize the data set, but you can do this step at the classification object
    """
    visualizeTrainData_for_SpecificRegression(routes_dataset1[1], routes_dataset1)














