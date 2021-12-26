 
from utils import FAKE_ITEM, unique
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from implicit.nearest_neighbours import bm25_weight, tfidf_weight
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class RecommenderDataset:
    """Базовый класс для удобства работы с данными
    Input
    -----
    data: pd.DataFrame
        DataFrame описывающий транзакции между user / item
    values: str
        поле по котором будет происходит агрегация и сортировка (сила взаимодействия user/item)
        по умолчанию `weight`
    aggfunc: str
        способ агрегации поля values (по умолчанию `mean`)
    weighting: str
        способ взвешивания матрицы user_item (`bm25` или `tfidf`)
        по умолчанию `bm25`
    """

    def __init__(self, data, values='weight', aggfunc='count', weighting='bm25'):

        self.users_top = data[data['item_id'] != FAKE_ITEM].groupby(
            ['user_id', 'item_id']).agg({values: aggfunc}).reset_index()
        self.users_top.sort_values(values, ascending=False, inplace=True)

        self.top = data[data['item_id'] != FAKE_ITEM].groupby('item_id').agg({values: aggfunc}).reset_index()
        self.top = self.top.sort_values(values, ascending=False).item_id.tolist()

        user_item_matrix = pd.pivot_table(data, index='user_id', columns='item_id',
                                          values=values, aggfunc=aggfunc,
                                          fill_value=0).astype(float)

        self.userids = user_item_matrix.index.values
        self.itemids = user_item_matrix.columns.values
        matrix_userids = np.arange(len(self.userids))
        matrix_itemids = np.arange(len(self.itemids))

        self.id_to_itemid = dict(zip(matrix_itemids, self.itemids))
        self.id_to_userid = dict(zip(matrix_userids, self.userids))
        self.itemid_to_id = dict(zip(self.itemids, matrix_itemids))
        self.userid_to_id = dict(zip(self.userids, matrix_userids))

        self.FAKE_ITEM_ID = self.itemid_to_id[FAKE_ITEM]
        self.__userTop = {}

        if weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(user_item_matrix.T).T
        elif weighting == 'bm25':
            self.user_item_matrix = bm25_weight(user_item_matrix.T).T
        else:
            self.user_item_matrix = user_item_matrix

        self.csr_matrix = csr_matrix(self.user_item_matrix).T.tocsr()

    def userExist(self, userId):
        return self.userid_to_id.get(userId, None) is not None

    def userTop(self, userId, N=50):
        if not self.userExist(userId):
            return []

        if self.__userTop.get(userId, None) is None:
            self.__userTop[userId] = self.users_top[self.users_top.user_id == userId].head(200).item_id.tolist()

        return self.__userTop[userId][:N]

    def extend(self, res, N=5):
        """дополняет результат до нужного количества из TOP товаров"""
        res = unique(list(res))

        if len(res) < N:
            res.extend(self.top[:N])
            res = unique(res)
            res = res[:N]

        assert len(res) == N, f'Количество рекомендаций != {N}'
        return res


class BaseRecommender:
    """Базовый класс для наследования
       Модель, которая всегда рекомендует полулярные товары
    Input
    -----
    ds: RecommenderDataset
        подготовленный RecommenderDataset обьект
    """

    def __init__(self, ds):
        # assert isinstance(ds, RecommenderDataset), 'Нужен обьект типа RecommenderDataset'
        self.ds = ds

    def fit(self):
        return self

    def extend(self, res, N=5, userId=None):
        res = list(res)
        if FAKE_ITEM in res:
            res.remove(FAKE_ITEM)

        if (len(res) < N) and (userId is not None):
            res.extend(self.ds.userTop(userId, N))
            res = unique(res)
            res = res[:N]
        return self.ds.extend(res, N)

    def _recommend(self, userId=None, N=5):
        return self.extend([], N=N, userId=userId)

    def _similarItems(self, userId=None, N=5):
        return self._recommend(userId, N)

    def _similarUsers(self, userId=None, N=5):
        return self._recommend(userId, N)

    def recommend(self, users, N=5, by='model'):
        if by == 'similarItems':
            func = self._similarItems
        elif by == 'similarUsers':
            func = self._similarUsers
        else:
            func = self._recommend
        return [func(user, N) for user in users]


class OwnRecommender(BaseRecommender):
    """Модель, которая рекомендует товары, среди товаров, купленных юзером
    Input
    -----
    ds: RecommenderDataset
        подготовленный RecommenderDataset обьект
    """

    def fit(self):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
        self.model = ItemItemRecommender(K=1, num_threads=4)
        self.model.fit(self.ds.csr_matrix)

        return self

    def _recommend(self, userId, N=5):
        """Рекомендуем товары для пользователя обученной моделью"""
        if not self.ds.userExist(userId):
            return self.ds.extend([], N)

        params = {
            'userid': self.ds.userid_to_id[userId],
            'user_items': self.ds.csr_matrix,
            'N': N,
            'filter_already_liked_items': False,
            'filter_items': [self.ds.FAKE_ITEM_ID],
            'recalculate_user': True
        }

        res = [
            self.ds.id_to_itemid[rec[0]]
            for rec in self.model.recommend(**params)
        ]

        return self.extend(res, N)


class AlsRecommender(OwnRecommender):
    """Модель, обученная ALS
    Input
    -----
    ds: RecommenderDataset
        подготовленный RecommenderDataset обьект
    """

    def fit(self, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        self.model = AlternatingLeastSquares(factors=n_factors,
                                             regularization=regularization,
                                             iterations=iterations,
                                             num_threads=num_threads)
        self.model.fit(self.ds.csr_matrix)

        return self

    def _similarItems(self, userId, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        if not self.ds.userExist(userId):
            return self.ds.extend([], N)

        def _get_similar_item(item_id):
            """Находит товар, похожий на item_id"""
            recs = self.model.similar_items(self.ds.itemid_to_id[item_id], N=2)
            if len(recs) > 1:
                top_rec = recs[1][0]
                return self.ds.id_to_itemid[top_rec]
            return item_id

        res = [_get_similar_item(item) for item in self.ds.userTop(userId, N)]
        return self.extend(res, N)

    def _similarUsers(self, userId, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        if not self.ds.userExist(userId):
            return self.ds.extend([], N)

        res = []
        similar_users = [rec[0] for rec in self.model.similar_users(self.ds.userid_to_id[userId], N=N+1)]
        similar_users = similar_users[1:]

        for user in similar_users:
            res.extend(self.ds.userTop(userId, 1))

        return self.extend(res, N)

    def items_embedings(self):
        emb = pd.DataFrame(data=self.model.item_factors).add_prefix('itm')
        emb['item_id'] = self.ds.itemids
        return emb

    def users_embedings(self):
        emb = pd.DataFrame(data=self.model.user_factors).add_prefix('usr')
        emb['user_id'] = self.ds.userids
        return emb


class Level2Recommender():
    """Модель ранжирования кандидатов на основе LGBMClassifier
    Input
    -----
        transactions: взаимодействия users vs items
        items: фичи items
        users: фичи users
    """

    def __init__(self, transactions, items=None, users=None, items_emb=None, users_emb=None):
        self._set_items(transactions, items, items_emb)
        self._set_users(transactions, users, users_emb)
        self.fitDone = False

    def _set_items(self, data, items=None, ie=None):
        istat = data.groupby('item_id').agg({
            'store_id': 'std',
            'basket_id': 'std',
            'sales_value': 'std',
            'quantity': 'std',
            'day': 'std',
        }).reset_index().add_prefix('i_')

        item_id = istat.i_item_id
        istat = 1/istat
        istat['item_id'] = item_id
        self.items = istat.drop('i_item_id', axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)

        if items is not None:
            self.items = self.items.merge(items, how='left', on='item_id')
        if ie is not None:
            item_features = self.items.merge(ie, how='outer', on='item_id')
            le = ie.shape[1]-1
            lx = len(item_features[item_features['itm0'].isnull()])
            embs = [list(ie[ie['item_id'] == 9999999].iloc[:, :-1].values[0])] * lx
            item_features.loc[item_features['itm0'].isnull(), -le:] = embs
            self.items = item_features

    def _set_users(self, data, users=None, ue=None):
        stat = data.groupby('user_id').agg({
            'store_id': 'nunique',
            'basket_id': 'nunique',
            'sales_value': 'mean',
            'quantity': 'mean',
            'coupon_disc': 'mean',
            'retail_disc': 'mean',
            'coupon_match_disc': 'mean',
            'trans_time': 'mean',
            'day': 'max',
        }).reset_index()

        stat['store_id'] /= data['store_id'].nunique()
        stat['basket_id'] /= data['basket_id'].nunique()
        stat['day'] /= data['day'].max()
        stat['trans_time'] /= data['trans_time'].max()
        self.users = stat.replace([np.inf, -np.inf], np.nan).fillna(0)

        if users is not None:
            self.users = self.users.merge(users, how='left', on='user_id')
        if ue is not None:
            self.users = self.users.merge(ue, how='outer', on='user_id')

    def _prepare_features(self, data):
        data = data[['user_id', 'item_id']]

        if self.items is not None:
            data = data.merge(self.items, how='left', on='item_id')

        if self.users is not None:
            data = data.merge(self.users, how='left', on='user_id')

        nll = data.isnull().sum()
        nll = nll[nll > 0]
        for i in nll.index:
            tp = str(data[i].dtype)
            vl = data[i].mode()[0] if tp == 'category' else data[i].mean()
            data[i].fillna(vl, inplace=True)

        return data

    def fit_predict_report(self, df):
        le = len(df) // 3
        tmp = df.sample(frac=1)
        train_set, test_set = tmp[:-le], tmp[-le:]

        X = train_set[['user_id', 'item_id']]
        y = train_set['target']
        self.fit(X, y)
        y_pred = self.predict(X)
        print("Качество на TRAIN:")
        print(classification_report(y, y_pred))

        X = test_set[['user_id', 'item_id']]
        y = test_set['target']
        y_pred = self.predict(X)
        print("Качество на TEST:")
        print(classification_report(y, y_pred))

    def fit(self, X, y, **kwargs):
        self.fitDone = False

        print('Fitting ...', end='')
        X = self._prepare_features(X)
        self.nums = X.dtypes[X.dtypes == float].index.tolist()
        self.cats = X.dtypes[X.dtypes == 'category'].index.tolist()
        self.cols = self.nums + self.cats

        # Скейлим числовые признаки
        self.sc = StandardScaler()
        X[self.nums] = self.sc.fit_transform(X[self.nums])

        # Убираем дисбаланс классов
        # cf = [X[self.cols].dtypes == 'category']
        # X, y = SMOTENC(random_state=42, sampling_strategy=0.3,
        #                categorical_features=cf).fit_resample(X[self.cols], y)
        X = pd.DataFrame(X, columns=self.cols)
        X[self.cats] = X[self.cats].astype('category')
        X[self.nums] = X[self.nums].astype(float)

        self.model = LGBMClassifier(objective='binary', n_estimators=300, reg_lambda=2, random_state=42)
        self.model.fit(X, y)
        print('done')
        self.fitDone = True
        return self

    def predict(self, X):
        if not self.fitDone:
            raise Exception('Fit is not done')
        X = self._prepare_features(X)
        X[self.nums] = self.sc.transform(X[self.nums])
        return self.model.predict(X[self.cols])

    def predict_proba(self, X):
        if not self.fitDone:
            raise Exception('Fit is not done')
        X = self._prepare_features(X)
        X[self.nums] = self.sc.transform(X[self.nums])
        return self.model.predict_proba(X[self.cols])[:, 0]

    def recommend(self, df, N=50):
        X = df[['user_id', 'item_id']].copy()
        X['predict'] = self.predict_proba(X)
        X = X.sort_values('predict', ascending=False).groupby('user_id')['item_id'].unique().reset_index()
        X['recommend_level2'] = X['item_id'].apply(lambda x: list(x)[:N])
        return X[['user_id', 'recommend_level2']]

    # def recommend(self, df, N=50):
    #     res = df.groupby('user_id')['item_id'].unique().reset_index()
    #     res.columns = ['user_id', 'actual']

    #     X = df[['user_id', 'item_id']].copy()
    #     X['predict'] = self.predict_proba(X)
    #     X = X.sort_values('predict', ascending=False).groupby('user_id')['item_id'].unique().reset_index()
    #     X['actual'] = X['actual'].apply(lambda x: list(x))
    #     X['recommend_level2'] = X['item_id'].apply(lambda x: list(x)[:N])

    #     return res.merge(X[['user_id', 'predict']], how='left', on='user_id')
