from dis import dis
from math import dist
from xml import dom

from sympy import re
import acquire
import pandas as pd
import prepare
import sklearn
import split
from geopy import distance

def drop_zillows_cols_and_na(df):
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    
    cols_to_drop = ['assessmentyear', 'roomcnt', 'id', 'calculatedbathnbr', 'finishedsquarefeet12', 'propertycountylandusecode',
                'id.1', 'propertylandusetypeid', 'regionidzip', 'propertylandusedesc', 'regionidcity', 'regionidcounty']

    df = df.drop(columns=cols_to_drop)
    df = df.dropna()

    return df


def rename_zillow(df):
    # readability
    df = df.rename(columns={'calculatedfinishedsquarefeet': 'sqr_ft'})
    return df

def threshold(df, index_col, prop_required):
    threshold = eval(f'int(round(prop_required*len(df.{index_col}),0))')
    #print(threshold)
    
    return threshold

def iqr_outlier(df):
    outliers = {}
    for col in df.columns:
        if df[col].dtype == 'float64':
            quartiles = pd.DataFrame(df[col].quantile([0.25,0.5,0.75]))
            #print(quartiles.iloc[0, 0])
            iqr = quartiles.iloc[2, 0] - quartiles.iloc[0, 0]
            upper_outlier_val = (iqr * 1.5) + quartiles.iloc[2,0]
            lower_outlier_val = quartiles.iloc[0,0] - (iqr * 1.5)

            outliers[col] = df[col][(df[col] < lower_outlier_val) & (df[col] > upper_outlier_val)]
    
    return outliers

def handle_missing_values(df, prop_required_column = .7, prop_required_row = .5):
    # create thresholds
    thresh = threshold(df, 'index', prop_required_column)
    
    # drop cols that don't meet prop requirement
    df.dropna(axis=1, thresh=thresh, inplace=True)
    
    thresh = threshold(df, 'columns', prop_required_row)
    # drop rows that don't meet prop requirement
    df.dropna(axis=0, thresh=thresh, inplace=True)
    
    # return changed dataframe with data that meets requirements
    return df

def domain_assumptions(df):
    # limit houses to include only >= 70 sqr ft 
    # (most prevelant minimum required sqr ft by state)
    df = df[df.sqr_ft >= 70]

    # exclude houses with bthroom/bedroomcnts of 0
    df = df[df.bedroomcnt != 0]
    df = df[df.bathroomcnt != 0.0]
    df = clean_sqr_feet(df)

    return df

def convert_zillow_dtypes(df):
    
    non_quant_ints = ['parcelid', 'latitude', 'longitude', 'censustractandblock',
                      'rawcensustractandblock']

    floats = []
    non_numeric = ['transactiondate', 'county']
    for col in df.columns:
        if col in non_quant_ints:
            df[col] = df[col].astype('int64')
        if col in non_numeric:
            df[col] = df[col].astype('object')
            if col == 'transactiondate':
                df[col] = pd.to_datetime(df[col])
        if col not in non_quant_ints and col not in non_numeric:
            df[col] = df[col].astype('float64')
            floats.append(col)

    return df, floats

def drop_duplicates(df):
    return df.sort_values('transactiondate').drop_duplicates('parcelid',keep='last')

def rename_cols(df):
    df = df.rename(columns={'calculatedfinishedsquarefeet': 'sqr_ft'})
    return df

def clean_zillow(df):
    # drop nulls and extra column
    #df = df.dropna()
    df = handle_missing_values(df)
    df = drop_zillows_cols_and_na(df)
    df = rename_cols(df)
    df = domain_assumptions(df)
    df = map_counties(df)
    df, quants = convert_zillow_dtypes(df)
    df = drop_duplicates(df)
    
    #remove numeric values with > 3.5 std dev
    df = prepare.remove_outliers(3.5, quants, df)
    cats = ['county']

    return df, quants, cats

def minimum_sqr_ft(df):
    #print(df)
    # min square footage for type of room
    bathroom_min = 10
    bedroom_min = 70
    
    # total MIN sqr feet
    total = df.bathroomcnt * bathroom_min + df.bedroomcnt * bedroom_min

    # return MIN sqr feet
    return total

def clean_sqr_feet(df):
    # get MIN sqr ft
    min_sqr_ft = minimum_sqr_ft(df)

    # return df with sqr_ft >= min_sqr_ft
    # change 'sqr_ft' to whichever name you have for sqr_ft in df
    return df[df.sqr_ft >= min_sqr_ft]

def map_counties(df):

    df.fips = df.fips.astype('object')

    # identified counties for fips codes 
    counties = {6037: 'los_angeles',
                6059: 'orange_county',
                6111: 'ventura'}

    # map counties to fips codes
    df.fips = df.fips.map(counties)

    # rename fips to county for clarity
    df.rename(columns=({ 'fips': 'county'}), inplace=True)

    return df

def wrangle_zillow():
    """ acquire and clean zillow data, returns zillow df, categorical data,
    and quantitative column names"""
    # aquire zillow data from mysql or csv
    zillow = acquire.get_zillow_data()

    # clean zillow data
    zillow, quants, cats = clean_zillow(zillow)


    return zillow, quants, cats

"train, test, validate"
def xy_tvt_data(train, validate, test, target_var):
    #for col in train.columns:
    #    if tr
    cols_to_drop = ['latitude', 'longitude', 
                    'parcelid', 'Unnamed: 0']

    
    x_train = train.drop(columns=drop_cols(cols_to_drop, 
                                           train, 
                                           target_var))
    y_train = train[target_var]


    x_validate = validate.drop(columns=drop_cols(cols_to_drop, 
                                              validate, 
                                              target_var))
    y_validate = validate[target_var]


    X_test = test.drop(columns=drop_cols(cols_to_drop, 
                                         test, 
                                         target_var))
    Y_test = test[target_var]

    return x_train, y_train, x_validate, y_validate, X_test, Y_test

def drop_cols(cols_to_drop, tvt_set, target_var):
    tvt_cols = [col for col in cols_to_drop if col in tvt_set.columns]
    tvt_cols.append(target_var)
    
    return tvt_cols

def encode_object_columns(train_df, drop_encoded=True):
    
    col_to_encode = object_columns_to_encode(train_df)
    #print(train_df.county)
    #print(col_to_encode)

    dummy_df = pd.get_dummies(train_df,
                              columns=col_to_encode,
                              dummy_na=False,
                              drop_first=True)
                            
    #print(dummy_df.head())
    train_df = pd.concat([train_df, dummy_df], axis=1)
    
    #print(train_df)
    if drop_encoded:
        train_df = drop_encoded_columns(train_df, col_to_encode)

    return train_df

def object_columns_to_encode(train_df):
    object_type = []
    #print(train_df.county.value_counts())
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            object_type.append(col)

    return object_type

def drop_encoded_columns(train_df, col_to_encode):
    train_df = train_df.drop(columns=col_to_encode)
    return train_df

def encoded_xy_data(train, validate, test, target_var):
    xy_train_validate_test = list(xy_tvt_data(train, validate, 
                                              test, target_var))
    
    print(xy_train_validate_test[0])
    for i in range(0, len(xy_train_validate_test), 2):
        
        xy_train_validate_test[i] = encode_object_columns(xy_train_validate_test[i])

    xy_train_validate_test = tuple(xy_train_validate_test)

    return xy_train_validate_test


def fit_and_scale(scaler, sets_to_scale):
    scaled_data = []
   # print(sets_to_scale[0].columns)
   # print(sets_to_scale[0][sets_to_scale[0].select_dtypes(include=['float64', 'uint8']).columns])
    scaler.fit(sets_to_scale[0][sets_to_scale[0].select_dtypes(include=['float64']).columns])
    print(sets_to_scale[0])

    for i in range(0, len(sets_to_scale), 1):
        #print(sets_to_scale[i].info())
        if i % 2 == 0:
            # only scales float columns
            floats = sets_to_scale[i].select_dtypes(include=['float64']).columns

            # fits scaler to training data only, then transforms 
            # train, validate & test
            scaled_data.append(pd.DataFrame(data=scaler.transform(sets_to_scale[i][floats]), columns=floats))
        else:
            scaled_data.append(sets_to_scale[i])


    return tuple(scaled_data)

def encoded_and_scaled(train, validate, test, target_var):
    sets_to_scale = encoded_xy_data(train, validate, test, target_var)

    #print(sets_to_scale)
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaled_data = fit_and_scale(scaler, sets_to_scale)

    return scaled_data

def rename_and_add_scaled_data(train, validate, test,
                               x_train_scaled, 
                               x_validate_scaled,
                               x_test_scaled):
    columns = {}

    for col in train.columns:
        if train[col].dtype == 'float64':
            columns[col] = f'scaled_{col}'

    x_train_scaled = x_train_scaled.rename(columns=columns)
    x_validate_scaled = x_validate_scaled.rename(columns=columns)
    x_test_scaled = x_test_scaled.rename(columns=columns)

    train = pd.concat([train.reset_index(), x_train_scaled], axis=1)
    validate = pd.concat([validate.reset_index(), x_validate_scaled], axis=1)
    test = pd.concat([test.reset_index(), x_test_scaled], axis=1)

    return train, validate, test, x_train_scaled, \
           x_validate_scaled, x_test_scaled


def distance_from_la(latitude, longitude):
    downtown_la_coords = [34.0488, -118.2518]
    return distance.geodesic(downtown_la_coords,[latitude, longitude]).km

            #return distance.distance(downtown_la_coords, [latitude, longitude])
def distance_from_santa_monica(latitude, longitude):
    santa_monica = [34.0195, -118.4912]
    return distance.geodesic(santa_monica,[latitude, longitude]).km


def distance_from_long_beach(latitude, longitude):
    long_beach_coords = [33.770050, -118.193741]    
    return distance.geodesic(long_beach_coords,[latitude, longitude]).km
        #return distance.distance(long_beach_coords, [latitude, longitude])

def distance_from_malibu(latitude, longitude):
    malibu = [34.0259, -118.7798]
    return distance.geodesic(malibu,[latitude, longitude]).km

def dist_from_bel_air(latitude, longitude):
    bel_air = [34.1002, -118.4595]
    return distance.geodesic(bel_air,[latitude, longitude]).km

def dist_balboa_island(latitude, longitude):
    balboa = [33.6073, -117.8971]
    return distance.geodesic(balboa, [latitude, longitude]).km

def dist_laguna_beach(latitude, longitude):
    laguna = [33.5427, -117.7854]
    return distance.geodesic(laguna, [latitude, longitude]).km

def dist_seal_beach(latitude, longitude):
    seal = [33.7414,-118.1048]
    return distance.geodesic(seal, [latitude, longitude]).km

def dist_channel_islands(latitude, longitude):
    channel = [34.1581,119.2232]
    return distance.geodesic(channel, [latitude, longitude]).km

def dist_ojai(latitude, longitude):
    ojai = [34.4480,-119.2429]
    return distance.geodesic(ojai, [latitude, longitude]).km

def dist_eleanor_sent(latitude, longitude):
    eleanor = [34.1354, -118.8568]
    return distance.geodesic(eleanor, [latitude, longitude]).km

def dist_ventura(latitude, longitude):
    ventura = [34.2805, -119.2945]
    return distance.geodesic(ventura, [latitude, longitude]).km

def dist_simi_valley(latitude, longitude):
    simi = [34.2694, -118.7815]
    return distance.geodesic(simi, [latitude, longitude]).km

def add_dist_cols(df, county):
    la_dist = ['dist_from_la', 'dist_from_long_beach',
                 'dist_santa_monica', 'dist_from_malibu',
                 'dist_from_bel_air']

    oc_dist = ['dist_balboa_island', 'dist_laguna_beach',
               'dist_seal_beach']

    ventura_dist = ['dist_simi', 'dist_ojai', 'dist_eleanor',
                    'dist_ventura', 'dist_channel_islands']

    if county=='la' or county=='all':
        df['dist_from_la'] = df.apply(lambda x: distance_from_la(x.latitude, x.longitude), axis=1) 
        df['dist_from_long_beach'] = df.apply(lambda x: distance_from_long_beach(x.latitude, x.longitude), axis=1) 
        df['dist_santa_monica'] = df.apply(lambda x: distance_from_santa_monica(x.latitude, x.longitude), axis=1) 
        df['dist_from_malibu'] = df.apply(lambda x: distance_from_malibu(x.latitude, x.longitude), axis=1) 
        df['dist_from_bel_air'] = df.apply(lambda x: dist_from_bel_air(x.latitude, x.longitude), axis=1)
        #df['sum_la_dist'] = df.apply(lambda x: x[la_dist].sum(), axis=1)
        

    if county=='oc' or county=='all':
        df['dist_balboa_island'] = df.apply(lambda x: dist_balboa_island(x.latitude, x.longitude), axis=1)
        df['dist_laguna_beach'] = df.apply(lambda x: dist_laguna_beach(x.latitude, x.longitude), axis=1)
        df['dist_seal_beach'] = df.apply(lambda x: dist_seal_beach(x.latitude, x.longitude), axis=1)
        #df['sum_oc_dist'] = df.apply(lambda x: x[oc_dist].sum(), axis=1)
        

    if county=='vent' or county=='all':
        df['dist_simi'] = df.apply(lambda x: dist_simi_valley(x.latitude, x.longitude), axis=1)
        df['dist_ventura'] = df.apply(lambda x: dist_ventura(x.latitude, x.longitude), axis=1)
        df['dist_channel_islands'] = df.apply(lambda x: dist_channel_islands(x.latitude, x.longitude), axis=1)
        df['dist_ojai'] = df.apply(lambda x: dist_ojai(x.latitude, x.longitude), axis=1)
        df['dist_eleanor'] = df.apply(lambda x: dist_eleanor_sent(x.latitude, x.longitude), axis=1)
        #df['sum_ventura_dist'] = df.apply(lambda x: x[ventura_dist].sum(), axis=1)
        
    return df

def all_train_validate_test_data(df, target_var, county):
    train, validate, test = split.train_validate_test_split(df, target_var)
    train = add_dist_cols(train, county)
    validate = add_dist_cols(validate, county)
    test = add_dist_cols(test, county)

    x_train_scaled, y_train, \
    x_validate_scaled, y_validate, \
    x_test_scaled, y_test = encoded_and_scaled(train, validate, test, target_var)

    train, validate, \
    test, x_train_scaled, \
    x_validate_scaled, \
    x_test_scaled = rename_and_add_scaled_data(train,
                                                validate, test,
                                                x_train_scaled,
                                                x_validate_scaled,
                                                x_test_scaled)
    
    return train, validate, test, \
           x_train_scaled, y_train, \
           x_validate_scaled, y_validate, \
           x_test_scaled, y_test

