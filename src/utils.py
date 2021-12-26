import numpy as np
import pandas as pd
import re

FAKE_ITEM = 9999999


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def prepare_items(items):
    res = items.drop('curr_size_of_product', axis=1)
    flds = items.columns.drop(['curr_size_of_product', 'item_id'])
    res[flds] = res[flds].astype('category')
    return res


def prepare_users(users):
    def row(x):
        x['ageFrom'], _, x['ageTo'] = re.match(
            r'(\d+)([\+\-](\d+))?', x['age_desc']).groups()
        if x['ageTo'] is None:
            x['ageTo'] = 100

        if x['income_desc'] == 'Under 15K':
            x['incomeFrom'] = 0
            x['incomeTo'] = 15
        else:
            x['incomeFrom'], _, x['incomeTo'] = re.match(
                r'(\d+)([\+\-](\d+))?', x['income_desc']).groups()
            if x['incomeTo'] is None:
                x['incomeTo'] = int(x['incomeFrom']) + 100

        x['household_size_desc'] = int(
            x['household_size_desc'].replace('+', ''))

        try:
            x['kid_category_desc'] = int(
                x['kid_category_desc'].replace('+', ''))
        except ValueError:
            x['kid_category_desc'] = 0

        return x.drop(['age_desc', 'income_desc'])

    res = users.apply(lambda x: row(x), axis=1)
    flds = ['marital_status_code', 'homeowner_desc', 'hh_comp_desc']
    res[flds] = res[flds].astype('category')

    flds = ['household_size_desc', 'kid_category_desc',
            'ageFrom', 'ageTo', 'incomeFrom', 'incomeTo']
    res[flds] = res[flds].astype('float')
    return res


def unstack_user_item(data, field, actual):
    s = data.apply(lambda x: pd.Series(
        x[field]), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'item_id'
    s = data.join(s)
    s['target'] = s.apply(lambda x: x['item_id']
                          in x[actual], axis=1).astype(float)
    return s[['user_id', 'item_id', 'target']]


def split_train_val(data, level1=6, level2=3):
    train = data[data['week_no'] < data['week_no'].max() - (level1 + level2)]
    validate1 = data[(data['week_no'] >= data['week_no'].max() - (level1 + level2)) &
                     (data['week_no'] < data['week_no'].max() - (level2))]
    validate2 = data[data['week_no'] >= data['week_no'].max() - level2]

    return train, validate1, validate2


def prefilter_items(source, price=(1, 50), popular=(0.01, 0.5), products=None, top_n=None):
    data = source.copy()
    data['price'] = data['sales_value'] / data['quantity']
    data['price'].fillna(0, inplace=True)

    stat = data.groupby('item_id').agg({
        'user_id': 'nunique',
        'price': 'median',
        'quantity': 'sum',
        'week_no': 'max'
    }).reset_index()

    stat['share_unique_users'] = stat['user_id'] / data['user_id'].nunique()

    flt = stat['user_id'] == 0
    if price[0] is not None:
        flt |= stat['price'] < price[0]  # дешевая цена за единицу товара
    else:
        flt |= stat['price'] <= 0  # нулевая цена
    if price[1] is not None:
        flt |= stat['price'] > price[1]  # дорогие товары
    if popular[0] is not None:
        # самые непопулярные товары
        flt |= stat['share_unique_users'] < popular[0]
    if popular[1] is not None:
        # самые популярные товары
        flt |= stat['share_unique_users'] > popular[1]

    stat = stat[~flt]

    if products is not None:
        # fld, mn = 'sub_commodity_desc', 50
        fld, mn = 'department', 150
        pcats = products.groupby(fld)['item_id'].nunique()
        pcats = pcats[pcats > mn]
        pcats = products[products[fld].isin(pcats.index)][['item_id']]
        stat = stat.merge(pcats, how='inner', on='item_id')

    stat['quantity'] /= stat['quantity'].max()

    stat['weight'] = (stat['share_unique_users'] + 0.1 *
                      stat['quantity']) * stat['week_no']
    if top_n is not None:
        stat = stat.sort_values('weight', ascending=False).head(top_n)
    stat['weight'] /= stat['weight'].min()

    data = data.merge(stat[['item_id', 'weight']], how='left', on='item_id')
    data.loc[data['weight'].isnull(), 'item_id'] = FAKE_ITEM
    data.loc[data['weight'].isnull(), 'weight'] = 0.0

    return data


def new_user_features(data, user_features):
    '''
    Добавляет среднюю корзину и средний чек за неделю.
    '''

    new_user_features = user_features.merge(data, on='user_id', how='left')

    basket = new_user_features.groupby(
        ['user_id'])['sales_value'].sum().reset_index()
    baskets_qnt = new_user_features.groupby(
        'user_id')['basket_id'].count().reset_index()
    baskets_qnt.rename(columns={'basket_id': 'baskets_qnt'}, inplace=True)
    average_basket = basket.merge(baskets_qnt)
    average_basket['average_basket'] = average_basket.sales_value / \
        average_basket.baskets_qnt
    average_basket['sum_per_week'] = average_basket.sales_value / \
        new_user_features.week_no.nunique()
    average_basket = average_basket.drop(
        ['sales_value', 'baskets_qnt'], axis=1)
    user_features = user_features.merge(average_basket)

    return user_features


def new_item_features(data, item_features):
    '''
    Считает цену, количество продаж и продаваемость за неделю.
    '''

    new_item_features = item_features.merge(data, on='item_id', how='left')

    price = new_item_features.groupby('item_id')['sales_value'].sum(
    ) / new_item_features.groupby('item_id')['quantity'].sum()
    price = price.groupby('item_id').mean().reset_index()
    price.columns = ['item_id', 'price']
    price['price'].fillna(0, inplace=True)

    item_qnt = new_item_features.groupby(
        ['item_id'])['quantity'].count().reset_index()
    item_qnt.rename(columns={'quantity': 'quantity_of_sales'}, inplace=True)
    item_qnt['quantity_of_sales_per_week'] = item_qnt['quantity_of_sales'] / \
        new_item_features['week_no'].nunique()

    item_features = item_features.merge(item_qnt, on='item_id')
    item_features = item_features.merge(price, on='item_id')
    item_features = item_features[item_features['quantity_of_sales'] > 0]

    return item_features


def postfilter_items():
    pass
