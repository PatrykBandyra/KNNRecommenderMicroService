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


def on_server_start():
    AdvancedRecommender()
