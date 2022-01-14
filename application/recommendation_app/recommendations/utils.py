import pandas as pd
import operator
from scipy import spatial
import pickle
import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


class AbstractRecommender:
    def __init__(self):
        self.users = pd.read_json("../../data/raw/users.jsonl", lines=True)
        self.sessions = pd.read_json("../../data/raw/sessions.jsonl", lines=True)
        self.products = pd.read_json("../../data/raw/products.jsonl", lines=True)

        self.category_list = self.products['category_path'].unique()
        self.products['category_name'] = self.products['category_path']
        self.products['category_path'] = self.products['category_path']. \
            apply(lambda x: SimpleRecommender.one_hot_encode(x, self.category_list))

        self.products['original_price'] = self.products['price']
        self.products['price'] = (self.products['price'] - self.products['price'].min()) / (
                self.products['price'].max() - self.products['price'].min())
        self.products['original_user_rating'] = self.products['user_rating']
        self.products['user_rating'] = (self.products['user_rating'] - self.products['user_rating'].min()) / (
                self.products['user_rating'].max() - self.products['user_rating'].min())

    def similarity(self, product_id1, product_id2):
        a = self.products.iloc[product_id1]
        b = self.products.iloc[product_id2]

        categoryA = a['category_path']
        categoryB = b['category_path']
        category_distance = spatial.distance.cosine(categoryA, categoryB)

        priceA = a['price']
        priceB = b['price']
        price_distance = abs(priceA - priceB) * 1

        ratingA = a['user_rating']
        ratingB = b['user_rating']
        rating_distance = abs(ratingA - ratingB)

        return category_distance + price_distance + rating_distance


class SimpleRecommender(AbstractRecommender):
    def __init__(self):
        super().__init__()

    @staticmethod
    def one_hot_encode(element, list):
        one_hot_encode_list = []

        for e in list:
            if element == e:
                one_hot_encode_list.append(1)
            else:
                one_hot_encode_list.append(0)
        return one_hot_encode_list

    def get_distances(self, product_id):
        p = self.products.index[self.products['product_id'] == product_id][0]
        distances = []

        for index, product in self.products.iterrows():
            if product['product_id'] != product_id:
                dist = self.similarity(index, p)
                distances.append((index, dist))

        distances.sort(key=operator.itemgetter(1))

        return distances

    def get_neighbours(self, product_id, K):
        """
        Throws IndexError if provided with invalid product id.
        """

        distances = self.get_distances(product_id)

        neighbours = []

        for x in range(K):
            neighbours.append(distances[x])
        return neighbours


class AdvancedRecommender(AbstractRecommender):
    filepath = os.path.join(os.path.dirname(__file__), 'all_distances.pk')

    def __init__(self):
        super().__init__()
        if os.path.exists(AdvancedRecommender.filepath):
            self.all_distances = self.load_all_distances()
        else:
            self.all_distances = self.get_all_distances()
            self.save_all_distances()

    def get_all_distances(self):
        all_distances = []
        for i in range(len(self.products)):
            all_distances.append(self.get_distances(self.products['product_id'].iloc[i]))
        return all_distances

    def save_all_distances(self):
        with open(AdvancedRecommender.filepath, 'wb') as f:
            pickle.dump(self.all_distances, f)

    def load_all_distances(self):
        with open(AdvancedRecommender.filepath, 'rb') as f:
            return pickle.load(f)

    def get_distances(self, product_id):
        p = self.products.index[self.products['product_id'] == product_id][0]
        distances = []

        for index, product in self.products.iterrows():
            if product['product_id'] != product_id:
                dist = self.similarity(index, p)
                distances.append(dist)
            else:
                distances.append(0)

        return distances

    def get_neighbours(self, distances, K):
        distances = [(index, dist) for index, dist in enumerate(distances)]
        distances.sort(key=operator.itemgetter(1))
        neighbours = []

        for x in range(K):
            neighbours.append(distances[x])
        return neighbours

    def get_recommendation(self, user_id, K):

        s = self.sessions.index[self.sessions['user_id'] == user_id].tolist()
        ids = []
        for i in s:
            ids.append(self.products.index[self.products['product_id'] == self.sessions['product_id'].iloc[i]][0])

        distances = [0] * len(self.products)
        for id in ids:
            distances = [sum(x) for x in zip(self.all_distances[id], distances)]

        maxi = max(distances)
        for id in ids:
            distances[id] = maxi

        return self.get_neighbours(distances, K)


class TheBestRecommender:
    def __init__(self):
        self.users = pd.read_json("../../data/raw/users.jsonl", lines=True)
        self.sessions = pd.read_json("../../data/raw/sessions.jsonl", lines=True)
        self.products = pd.read_json("../../data/raw/products.jsonl", lines=True)

        self.sessions['score'] = self.sessions['event_type'].map({'VIEW_PRODUCT': 5, 'BUY_PRODUCT': 5})

        self.group = self.sessions.groupby(['user_id', 'product_id'])['score'].sum().reset_index()
        self.group['score'] = self.group['score'].apply(lambda x: 5 if x > 5 else x)
        self.group = pd.pivot_table(self.group, values='score', index='user_id', columns='product_id')
        self.group = self.group.fillna(0)
        self.group = self.group.stack().reset_index()
        self.group = self.group.rename(columns={0: 'score'})
        self.group['user_view'] = self.group['score'].apply(lambda x: 1 if x > 0 else 0)

        self.std = MinMaxScaler(feature_range=(0, 1))
        self.std.fit(self.group['score'].values.reshape(-1, 1))
        self.group['interaction_score'] = self.std.transform(self.group['score'].values.reshape(-1, 1))

        self.group = pd.merge(self.group, self.products, on="product_id", how="left")
        self.group = pd.merge(self.group, self.users, on="user_id", how="left")
        self.group = self.group[
            ['user_id', 'product_id', 'product_name', 'category_path', 'price', 'user_rating', 'score',
             'interaction_score', 'user_view']]
        self.group['price'] = self.group['price'].apply(lambda x: TheBestRecommender.price_bin(x))
        self.group['user_rating'] = self.group['user_rating'].apply(lambda x: TheBestRecommender.rating_bin(x))

        self.matrix = pd.pivot_table(self.group, values='score', index='user_id', columns='product_id')
        self.matrix = self.matrix.fillna(0)

        self.product_cat = self.group[['product_id', 'category_path', 'price', 'user_rating']].drop_duplicates(
            'product_id')
        self.product_cat = self.product_cat.sort_values(by='product_id')

        self.price_matrix = np.reciprocal(euclidean_distances(np.array(self.product_cat['price']).reshape(-1, 1)) + 1)
        self.euclidean_matrix1 = pd.DataFrame(self.price_matrix, columns=self.product_cat['product_id'],
                                              index=self.product_cat['product_id'])

        self.rating_matrix = np.reciprocal(
            euclidean_distances(np.array(self.product_cat['user_rating']).reshape(-1, 1)) + 1)
        self.euclidean_matrix2 = pd.DataFrame(self.rating_matrix, columns=self.product_cat['product_id'],
                                              index=self.product_cat['product_id'])

        self.tfidf_vectorizer = TfidfVectorizer()
        self.doc_term = self.tfidf_vectorizer.fit_transform(list(self.product_cat['category_path']))
        self.dt_matrix = pd.DataFrame(self.doc_term.toarray().round(3),
                                      index=[i for i in self.product_cat['product_id']],
                                      columns=self.tfidf_vectorizer.get_feature_names())
        self.cos_similar_matrix = pd.DataFrame(cosine_similarity(self.dt_matrix.values),
                                               columns=self.product_cat['product_id'],
                                               index=self.product_cat['product_id'])

        self.similarity_matrix = self.euclidean_matrix1.multiply(self.euclidean_matrix2).multiply(
            self.cos_similar_matrix)
        self.content_matrix = self.matrix.dot(self.similarity_matrix)
        self.std = MinMaxScaler(feature_range=(0, 1))
        self.std.fit(self.content_matrix.values)
        self.content_matrix = self.std.transform(self.content_matrix.values)
        self.content_matrix = pd.DataFrame(self.content_matrix, columns=sorted(self.group['product_id'].unique()),
                                           index=sorted(self.group['user_id'].unique()))

        self.content_df = self.content_matrix.stack().reset_index()
        self.content_df = self.content_df.rename(
            columns={'level_0': 'user_id', 'level_1': 'product_id', 0: 'predicted_interaction'})

        self.group = self.group.merge(self.content_df, on=['user_id', 'product_id'])
        self.group['predicted_view'] = self.group['predicted_interaction'].apply(lambda x: 1 if x >= 0.5 else 0)

    @staticmethod
    def price_bin(price):
        if price <= 25:
            return 0
        if price <= 50:
            return 1
        if price <= 100:
            return 2
        if price <= 250:
            return 3
        if price <= 500:
            return 4
        if price <= 1000:
            return 5
        if price <= 2000:
            return 6
        if price <= 4000:
            return 7
        else:
            return 8

    @staticmethod
    def rating_bin(rating):
        if rating <= 0.5:
            return 0
        if rating <= 1.5:
            return 1
        if rating <= 2.5:
            return 2
        if rating <= 3.5:
            return 3
        if rating <= 4.5:
            return 4
        else:
            return 5

    def get_recommendation(self, user_id, K):
        s = self.sessions.index[self.sessions['user_id'] == user_id].tolist()
        ids = []
        for i in s:
            ids.append(self.sessions['product_id'].iloc[i])

        user_content_df = self.content_df.loc[self.content_df['user_id'] == user_id]

        mini = user_content_df['predicted_interaction'].min()
        for id in ids:
            user_content_df.loc[user_content_df['product_id'] == id, 'predicted_interaction'] = mini

        user_content_df = user_content_df.sort_values(by="predicted_interaction", ascending=False)

        recommendations = []
        for x in range(K):
            recommendations.append(user_content_df['product_id'].iloc[x])
        return recommendations


class ExperimentAB:
    filepath = os.path.join(os.path.dirname(__file__), 'all_distances.pk')
    SEED = 32
    np.random.seed(SEED)

    def __init__(self):
        self.users = pd.read_json("../../data/raw/users.jsonl", lines=True)
        self.sessions = pd.read_json("../../data/raw/sessions.jsonl", lines=True)
        self.products = pd.read_json("../../data/raw/products.jsonl", lines=True)

        if os.path.exists(AdvancedRecommender.filepath):
            self.all_distances = self.load_all_distances()
        else:
            self.all_distances = self.get_all_distances()
            self.save_all_distances()

        # Model A --------------------------------------------------------------------
        self.sessionsA = self.sessions.copy()

        self.sessionsA['score'] = self.sessionsA['event_type'].map({'VIEW_PRODUCT': 1, 'BUY_PRODUCT': 0})

        self.groupA = self.sessionsA.groupby(['user_id', 'product_id'])['score'].sum().reset_index()

        self.groupA = pd.pivot_table(self.groupA, values='score', index='user_id', columns='product_id')
        self.groupA = self.groupA.fillna(0)
        self.groupA = self.groupA.stack().reset_index()
        self.groupA = self.groupA.rename(columns={0: 'score'})
        self.groupA['user_view'] = self.groupA['score'].apply(lambda x: 1 if x > 0 else 0)

        mask = np.random.rand(len(self.groupA)) < 0.8
        self.trainsetA = self.groupA[mask]
        self.testsetA = self.groupA[~mask]
        self.trainsetA = self.trainsetA.reset_index()

        self.category_list = self.products['category_path'].unique()

        self.productsA = self.products.copy()
        self.productsA['category_path'] = self.productsA['category_path'].apply(
            lambda x: self.one_hot_encode(x, self.category_list))
        self.productsA['price'] = (self.productsA['price'] - self.productsA['price'].min()) / (
                self.productsA['price'].max() - self.productsA['price'].min())
        self.productsA['user_rating'] = (self.productsA['user_rating'] - self.productsA['user_rating'].min()) / (
                self.productsA['user_rating'].max() - self.productsA['user_rating'].min())

        self.user_mask = np.random.rand(len(self.users)) < 0.2
        self.test_users = self.users[self.user_mask]

        # Model B --------------------------------------------------------------------
        self.sessionsB = self.sessions.copy()
        self.sessionsB['score'] = self.sessionsB['event_type'].map({'VIEW_PRODUCT': 5, 'BUY_PRODUCT': 5})

        self.groupB = self.sessionsB.groupby(['user_id', 'product_id'])['score'].sum().reset_index()

        self.groupB = self.sessionsB.groupby(['user_id', 'product_id'])['score'].sum().reset_index()
        self.groupB['score'] = self.groupB['score'].apply(lambda x: 5 if x > 5 else x)
        self.groupB = pd.pivot_table(self.groupB, values='score', index='user_id', columns='product_id')
        self.groupB = self.groupB.fillna(0)
        self.groupB = self.groupB.stack().reset_index()
        self.groupB = self.groupB.rename(columns={0: 'score'})
        self.groupB['user_view'] = self.groupB['score'].apply(lambda x: 1 if x > 0 else 0)

        self.std1 = MinMaxScaler(feature_range=(0, 1))
        self.std1.fit(self.groupB['score'].values.reshape(-1, 1))
        self.groupB['interaction_score'] = self.std1.transform(self.groupB['score'].values.reshape(-1, 1))

        self.groupB = pd.merge(self.groupB, self.products, on="product_id", how="left")
        self.groupB = pd.merge(self.groupB, self.users, on="user_id", how="left")
        self.groupB = self.groupB[['user_id', 'product_id', 'product_name', 'category_path', 'price', 'user_rating',
                                   'score', 'interaction_score', 'user_view']]
        self.groupB['price'] = self.groupB['price'].apply(lambda x: ExperimentAB.price_bin(x))
        self.groupB['user_rating'] = self.groupB['user_rating'].apply(lambda x: ExperimentAB.rating_bin(x))

        self.trainsetB = self.groupB[mask]
        self.testsetB = self.groupB[~mask]
        self.trainsetB = self.trainsetB.reset_index()

        self.train_matrix = pd.pivot_table(self.trainsetB, values='score', index='user_id', columns='product_id')
        self.train_matrix = self.train_matrix.fillna(0)

        self.product_cat = self.trainsetB[['product_id', 'category_path', 'price', 'user_rating']].drop_duplicates(
            'product_id')
        self.product_cat = self.product_cat.sort_values(by='product_id')

        self.price_matrix = np.reciprocal(euclidean_distances(np.array(self.product_cat['price']).reshape(-1, 1)) + 1)
        self.euclidean_matrix1 = pd.DataFrame(self.price_matrix, columns=self.product_cat['product_id'],
                                              index=self.product_cat['product_id'])

        self.rating_matrix = np.reciprocal(
            euclidean_distances(np.array(self.product_cat['user_rating']).reshape(-1, 1)) + 1)
        self.euclidean_matrix2 = pd.DataFrame(self.rating_matrix, columns=self.product_cat['product_id'],
                                              index=self.product_cat['product_id'])

        self.tfidf_vectorizer = TfidfVectorizer()
        self.doc_term = self.tfidf_vectorizer.fit_transform(list(self.product_cat['category_path']))
        self.dt_matrix = pd.DataFrame(self.doc_term.toarray().round(3),
                                      index=[i for i in self.product_cat['product_id']],
                                      columns=self.tfidf_vectorizer.get_feature_names())
        self.cos_similar_matrix = pd.DataFrame(cosine_similarity(self.dt_matrix.values),
                                               columns=self.product_cat['product_id'],
                                               index=self.product_cat['product_id'])

        self.similarity_matrix = self.euclidean_matrix1.multiply(self.euclidean_matrix2).multiply(
            self.cos_similar_matrix)
        self.content_matrix = self.train_matrix.dot(self.similarity_matrix)
        self.std2 = MinMaxScaler(feature_range=(0, 1))
        self.std2.fit(self.content_matrix.values)
        self.content_matrix = self.std2.transform(self.content_matrix.values)
        self.content_matrix = pd.DataFrame(self.content_matrix, columns=sorted(self.trainsetB['product_id'].unique()),
                                           index=sorted(self.trainsetB['user_id'].unique()))

        self.content_df = self.content_matrix.stack().reset_index()
        self.content_df = self.content_df.rename(
            columns={'level_0': 'user_id', 'level_1': 'product_id', 0: 'predicted_interaction'})

    def get_recommendationB(self, user_id, K):
        s = self.trainsetB.index[self.trainsetB['user_id'] == user_id].tolist()
        ids = []
        for i in s:
            ids.append(self.trainsetB['product_id'].iloc[i])

        user_content_df = self.content_df.loc[self.content_df['user_id'] == user_id]

        mini = user_content_df['predicted_interaction'].min()
        for id in ids:
            user_content_df.loc[user_content_df['product_id'] == id, 'predicted_interaction'] = mini

        user_content_df = user_content_df.sort_values(by="predicted_interaction", ascending=False)

        recommendations = []
        for x in range(K):
            recommendations.append(user_content_df['product_id'].iloc[x])
        return recommendations

    def get_accuracy_B_by_user_id(self, users, K):
        correct = 0
        for user in users:
            recommendations = self.get_recommendationB(user, K)
            for recommendation in recommendations:
                view = self.testsetB[(self.testsetB['product_id'] == recommendation) &
                                     (self.testsetB['user_id'] == user)]['user_view']
                if len(view) != 0:
                    view = view.item()
                    if view == 1:
                        correct = correct + 1
        return correct / (K * len(self.test_users))

    def get_accuracy_B(self, K):
        correct = 0
        for index, user in self.test_users.iterrows():
            recommendations = self.get_recommendationB(user['user_id'], 5)
            for recommendation in recommendations:
                view = self.testsetB[(self.testsetB['product_id'] == recommendation) &
                                     (self.testsetB['user_id'] == user['user_id'])]['user_view']
                if len(view) != 0:
                    view = view.item()
                    if view == 1:
                        correct = correct + 1
        return correct / (K * len(self.test_users))

    @staticmethod
    def price_bin(price):
        if price <= 25:
            return 0
        if price <= 50:
            return 1
        if price <= 100:
            return 2
        if price <= 250:
            return 3
        if price <= 500:
            return 4
        if price <= 1000:
            return 5
        if price <= 2000:
            return 6
        if price <= 4000:
            return 7
        else:
            return 8

    @staticmethod
    def rating_bin(rating):
        if rating <= 0.5:
            return 0
        if rating <= 1.5:
            return 1
        if rating <= 2.5:
            return 2
        if rating <= 3.5:
            return 3
        if rating <= 4.5:
            return 4
        else:
            return 5

    def get_accuracy_A_by_user_id(self, users, K):
        correct = 0
        for user in users:
            recommendations = self.get_recommendationA(user, K)
            for recommendation in recommendations:
                id = self.products.iloc[recommendation[0]]['product_id']
                view = self.testsetA[(self.testsetA['product_id'] == id) &
                                     (self.testsetA['user_id'] == user)]['user_view']
                if len(view) != 0:
                    view = view.item()
                    if view == 1:
                        correct = correct + 1
        return correct / (K * len(self.test_users))

    def get_accuracy_A(self, K):
        correct = 0
        for index, user in self.test_users.iterrows():
            recommendations = self.get_recommendationA(user['user_id'], K)
            for recommendation in recommendations:
                id = self.products.iloc[recommendation[0]]['product_id']
                view = self.testsetA[(self.testsetA['product_id'] == id) &
                                     (self.testsetA['user_id'] == user['user_id'])]['user_view']
                if len(view) != 0:
                    view = view.item()
                    if view == 1:
                        correct = correct + 1
        return correct / (K * len(self.test_users))

    def get_recommendationA(self, user_id, K):
        s = self.trainsetA.index[self.trainsetA['user_id'] == user_id].tolist()
        ids = []
        for i in s:
            ids.append(self.trainsetA['product_id'].iloc[i])

        distances = [0] * len(self.products)
        for id in ids:
            i = self.products.index[self.products['product_id'] == id][0]
            score = self.trainsetA.loc[(self.trainsetA['user_id'] == user_id) & (self.trainsetA['product_id'] == id),
                                       'score']
            if len(score) != 0:
                score = score.item()
                for p in range(len(self.products)):
                    distances[p] = distances[p] + self.all_distances[i][p] * score

        maxi = max(distances)
        for id in ids:
            distances[self.products.index[self.products['product_id'] == id][0]] = maxi

        return ExperimentAB.get_neighbours(distances, K)

    def similarity(self, product_id1, product_id2):
        a = self.productsA.iloc[product_id1]
        b = self.productsA.iloc[product_id2]

        categoryA = a['category_path']
        categoryB = b['category_path']
        category_distance = spatial.distance.cosine(categoryA, categoryB)

        priceA = a['price']
        priceB = b['price']
        price_distance = abs(priceA - priceB) * 1

        ratingA = a['user_rating']
        ratingB = b['user_rating']
        rating_distance = abs(ratingA - ratingB)

        return category_distance + price_distance + rating_distance

    @staticmethod
    def one_hot_encode(element, list):
        one_hot_encode_list = []

        for e in list:
            if element == e:
                one_hot_encode_list.append(1)
            else:
                one_hot_encode_list.append(0)
        return one_hot_encode_list

    def get_distancesA(self, product_id):
        p = self.products.index[self.products['product_id'] == product_id][0]
        distances = []

        for index, product in self.products.iterrows():
            if product['product_id'] != product_id:
                dist = self.similarity(index, p)
                distances.append(dist)
            else:
                distances.append(0)

        return distances

    @staticmethod
    def get_neighbours(distances, K):
        distances = [(index, dist) for index, dist in enumerate(distances)]
        distances.sort(key=operator.itemgetter(1))
        neighbours = []

        for x in range(K):
            neighbours.append(distances[x])
        return neighbours

    def get_all_distances(self):
        all_distances = []
        for i in range(len(self.products)):
            all_distances.append(self.get_distances(self.products['product_id'].iloc[i]))
        return all_distances

    def save_all_distances(self):
        with open(AdvancedRecommender.filepath, 'wb') as f:
            pickle.dump(self.all_distances, f)

    def load_all_distances(self):
        with open(AdvancedRecommender.filepath, 'rb') as f:
            return pickle.load(f)

    def get_distances(self, product_id):
        p = self.products.index[self.products['product_id'] == product_id][0]
        distances = []

        for index, product in self.products.iterrows():
            if product['product_id'] != product_id:
                dist = self.similarity(index, p)
                distances.append(dist)
            else:
                distances.append(0)

        return distances


def on_server_start():
    AdvancedRecommender()
