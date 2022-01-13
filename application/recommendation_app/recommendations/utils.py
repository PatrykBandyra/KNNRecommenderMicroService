import pandas as pd
import operator
from scipy import spatial
import pickle
import os


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


def on_server_start():
    AdvancedRecommender()
