import catboost
from src.metrics import precision_at_k
import pandas as pd
import numpy as np
import yaml
#from yaml.loader import FullLoader

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight
from catboost import Pool, CatBoostRanker

class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data: pd.DataFrame, weighting: bool = True, gpu=False, K1=100, B=0.8):
        _CONSTANTS = {
            'N_PREDICT': 50, # N = Neighbors
            'TOPK_PRECISION': 5,
            }
        
        self._GPU = gpu
        self._CONSTANTS = _CONSTANTS
        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])[
            'quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[
            self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = \
            self._prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T, K1=K1, B=B).T

        self.model = self.fit(self, self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    def _prepare_matrix(self, data: pd.DataFrame):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0,
                                          sort=False)#!

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def _prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
        own_recommender = ItemItemRecommender(K=1, num_threads=6)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        return own_recommender

    def recommend_1lvl(self, data_actual):
        data_actual['own_recs'] = data_actual['user_id'].apply(
            lambda x: self.get_own_recommendations(x, N=self._CONSTANTS['N_PREDICT']))

        data_actual['own_recs_score'] = data_actual.apply(
            lambda x: precision_at_k(x['own_recs'], x['actual'], k=self._CONSTANTS['TOPK_PRECISION']), axis=1).mean()

        return data_actual

    def preprocessing(self, dataset, t='train', training=True):
        """
        Prepare data for ranker -- splitting in correct way
        :param dataset: object of Dataset class
        :param t: 'train' (data_train_lvl_2) or 'valuate' (data_val_lvl_2)
        :param training: if use dataset.data_test - use True
        :return: X and y
        """

        if training:
            data_train_lvl_1 = dataset.data_train_lvl_1
            if t == 'train':
                df = dataset.data_train_lvl_2
            else:
                df = dataset.data_val_lvl_2
        else:
            data_train_lvl_1 = dataset.data_train_lvl_1_real
            if t == 'train':
                df = dataset.data_train_lvl_2_real
            else:
                df = dataset.data_val_lvl_2_real

        # creating dataset for ranking
        df_match_candidates = pd.DataFrame(df['user_id'].unique())
        df_match_candidates.columns = ['user_id']
        df_match_candidates = df_match_candidates[
            df_match_candidates['user_id'].isin(
                data_train_lvl_1['user_id'].unique())]
        df_match_candidates['candidates'] = df_match_candidates['user_id'].apply(
            lambda x: self.get_own_recommendations(x, N=self._CONSTANTS['N_PREDICT']))

        df_items = df_match_candidates.apply(lambda x: pd.Series(x['candidates']), axis=1) \
            .stack() \
            .reset_index(level=1, drop=True)
        df_items.name = 'item_id'
        df_match_candidates = df_match_candidates.drop('candidates', axis=1).join(df_items)

        # Создаем трейн сет для ранжирования с учетом кандидатов с этапа 1
        df_ranker_train = df[['user_id', 'item_id']].copy()
        df_ranker_train['target'] = 1  # тут только покупки
        df_ranker_train = df_match_candidates.merge(df_ranker_train,
                                                    on=['user_id', 'item_id'],
                                                    how='left')
        df_ranker_train['target'].fillna(0, inplace=True)

        # merging
        df_ranker_train = df_ranker_train.merge(dataset.item_features, on='item_id', how='left')
        df_ranker_train = df_ranker_train.merge(dataset.user_features, on='user_id', how='left')

        if training:
            if t == "train":
                # train split
                self.X_train = df_ranker_train.drop('target', axis=1)
                self.y_train = df_ranker_train[['target']]
            else:
                # test split
                self.X_test = df_ranker_train.drop('target', axis=1)
                self.y_test = df_ranker_train[['target']]
        else:
            if t == "train":
                # train split
                self.X_train_real = df_ranker_train.drop('target', axis=1)
                self.y_train_real = df_ranker_train[['target']]
            else:
                # test split
                self.X_test_real = df_ranker_train.drop('target', axis=1)
                self.y_test_real = df_ranker_train[['target']]

    @staticmethod
    def fit(self, user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads,
                                        use_gpu=self._GPU)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=None):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""
        if not N:
            N = self._CONSTANTS['N_PREDICT']

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=None):
        """Рекомендации через стардартные библиотеки implicit"""
        if not N:
            N = self._CONSTANTS['N_PREDICT']

        self._update_dict(user_id=user)
        if 999999 in self.itemid_to_id:
            filt = [self.itemid_to_id[999999]]
        else:
            filt = None
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                    user_items=csr_matrix(
                                                                        self.user_item_matrix).tocsr(),
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=filt,
                                                                    recalculate_user=True)]
        res = self._extend_with_top_popular(res, N=N)
        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=None):
        """Рекомендации через стардартные библиотеки implicit"""
        if not N:
            N = self._CONSTANTS['N_PREDICT']

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=None):
        """Рекомендуем товары среди тех, которые юзер уже купил"""
        if not N:
            N = self._CONSTANTS['N_PREDICT']

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_items_recommendation(self, user_id, N=None):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        if not N:
            N = self._CONSTANTS['N_PREDICT']

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user_id].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user_id, N=None):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        if not N:
            N = self._CONSTANTS['N_PREDICT']

        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user_id], N=N + 1)
        similar_users = [self.id_to_userid[rec[0]] for rec in similar_users]
        similar_users = similar_users[1:]  # удалим юзера из запроса

        for _user_id in similar_users:
            res.extend(self.get_own_recommendations(_user_id, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def ranker_fit(self, training=True):

        if training:
            X_train = self.X_train
            y_train = self.y_train
            X_test  = self.X_test
            y_test  = self.y_test
        else:
            X_train = self.X_train_real
            y_train = self.y_train_real
            X_test = self.X_test_real
            y_test = self.y_test_real
        # формируем фичи для обучения
        cat_feats = X_train.columns[2:].tolist()

        train_pool = Pool(data=X_train[cat_feats],
                          label=y_train,
                          group_id=X_train['user_id'])

        test_pool = Pool(data=X_test[cat_feats],
                         label=y_test,
                         group_id=X_test['user_id'])
        
        tt_PU = 'CPU'
        if self._GPU:
            tt_PU = 'GPU'

        parameters = {'iterations': 200, 
                      'silent': True,
                      'eta': 0.15, 
                      'task_type': tt_PU,
                      'max_depth': 7,
                      'loss_function': 'PairLogitPairwise',
                      'random_state': 29,
                      }

        ranker_model = CatBoostRanker(**parameters)
        ranker_model.fit(train_pool, eval_set=test_pool, plot=True)

        if training:
            self.ranker_model = ranker_model
        else:
            self.ranker_model_real = ranker_model

    def ranker_predict(self, df, training=True):
        if training:
            ranker_model = self.ranker_model
        else:
            ranker_model = self.ranker_model_real
        df['predict'] = catboost.CatBoost.predict(ranker_model,
                                                  df,
                                                  prediction_type='Probability')[:, 1]
        return df

    @staticmethod
    def rerank(df, user_id):
        return df[df['user_id'] == user_id].sort_values('predict', ascending=False).head(5).item_id.tolist()

    def evaluate_2models(self, training=True):

        if training:
             X_test = self.X_test
             y_test = self.y_test
        else:
            X_test = self.X_test_real
            y_test = self.y_test_real

        result_eval_ranker = X_test.groupby('user_id')['item_id'].unique().reset_index()
        result_eval_ranker.columns = ['user_id', 'actual']

        # get real target answers
        X_test_y = X_test.merge(y_test, right_index=True, left_index=True)
        y_test_unique = X_test_y[X_test_y['target'] == 1.0].groupby('user_id')['item_id'].unique().reset_index()
        y_test_unique.columns = ['user_id', 'y_actual']
        y_test_unique.head(2)

        # add y_test as y_actual
        result_eval_ranker = result_eval_ranker.merge(y_test_unique, on='user_id', how='left').fillna("").apply(list)

        # add probabilities
        X_test = self.ranker_predict(X_test, training=training)

        result_eval_ranker['own_rec'] = result_eval_ranker['user_id'].apply(
            lambda x: self.get_own_recommendations(x, N=self._CONSTANTS['N_PREDICT']))

        result_eval_ranker['ranked_own_rec'] = result_eval_ranker['user_id'].apply(
            lambda user_id: self.rerank(X_test, user_id))

        precision_ranked_matcher = result_eval_ranker['user_id'].apply(lambda x: precision_at_k(
            result_eval_ranker.loc[result_eval_ranker['user_id'] == x, 'ranked_own_rec'].squeeze(),
            result_eval_ranker.loc[result_eval_ranker['user_id'] == x, 'y_actual'].squeeze(),
            k=self._CONSTANTS["TOPK_PRECISION"])).mean()

        print(f'precision@{self._CONSTANTS["TOPK_PRECISION"]} of 2lvl-model is {precision_ranked_matcher}')

        if not training:
            result_eval_ranker.to_csv('result_test.csv', index=False)
            print('Файл с результатами сохранен.')