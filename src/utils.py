import numpy as np


def prefilter_items(data, item_features, n_popular=5000, first_sort_popular=True):
    if first_sort_popular:
            # Оставим только n самых популярных товаров перед остальными фильтрами
        popularity = data.groupby('item_id')['quantity'].count().reset_index() #### sum
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top_n_popular = popularity.sort_values('n_sold', ascending=False).\
                                        head(n_popular).item_id.tolist()
        #добавим, чтобы не потерять пользователей
        data.loc[~data['item_id'].isin(top_n_popular), 'item_id'] = 999999
        
    # Уберем самые популярные товары из списка - их не надо рекламировать 
    # (их и так покупают больше 20% пользователей)
    popularity = (data.groupby('item_id')['user_id'].nunique() / data.user_id.nunique()).reset_index() 
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity.share_unique_users > 0.2].item_id.tolist()
    data.loc[data['item_id'].isin(top_popular), 'item_id'] = 999999
    
    # Уберем самые НЕпопулярные товары из списка - их бесполезно рекомендовать
    unpopular = popularity[popularity.share_unique_users < 0.02].item_id.tolist()
    data.loc[data['item_id'].isin(unpopular), 'item_id'] = 999999 
    
    # Уберем товары, которые не продавались последние 12 месяцев
    t = data.week_no.max() - 52
    sold_last_12_month = data[data.week_no >= t].item_id.unique().tolist()
    data.loc[~data['item_id'].isin(sold_last_12_month), 'item_id'] = 999999
    
    # Уберем не интересные для рекоммендаций категории (department)
    department_size = item_features.groupby('department')['item_id'].nunique().\
                                       sort_values(ascending=False).reset_index()
    department_size.rename(columns={'item_id': 'n_items'}, inplace=True)
    slow_departments = department_size[department_size.n_items < 150].department.tolist()

    items_in_slow_deparmments = item_features[
        item_features.department.isin(slow_departments)].item_id.unique().tolist()
    data.loc[data['item_id'].isin(items_in_slow_deparmments), 'item_id'] = 999999
    
    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data['price'] = data['sales_value'] / np.maximum(data['quantity'], 1)
    data.loc[data['price'] < 2, 'item_id'] = 999999
    
    # Уберем слишком дорогие товарыs
    data.loc[data['price'] > 50, 'item_id'] = 999999
    
    if not first_sort_popular:
        # Оставим только n самых популярных товаров после фильтрации
        popularity = data.groupby('item_id')['quantity'].count().reset_index() #### sum
        popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top_n_popular = popularity.sort_values('n_sold', ascending=False).\
                                        head(n_popular).item_id.tolist()
        #добавим, чтобы не потерять пользователей
        data.loc[~data['item_id'].isin(top_n_popular), 'item_id'] = 999999
        
    return data




def postfilter_items():
    pass
