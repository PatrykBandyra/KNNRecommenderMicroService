import pandas as pd
import operator
from scipy import spatial


class SimpleRecommender:
    def __init__(self):

        self.users = pd.read_json("../../data/raw/users.jsonl", lines=True)
        self.sessions = pd.read_json("../../data/raw/sessions.jsonl", lines=True)
        self.products = pd.read_json("../../data/raw/products.jsonl", lines=True)

        self.category_list = self.products['category_path'].unique()
        self.products['category_name'] = self.products['category_path']
        self.products['category_path'] = self.products['category_path']. \
            apply(lambda x: SimpleRecommender.one_hot_encode(x, self.category_list))

        self.products['price'] = (self.products['price'] - self.products['price'].min()) / (
                self.products['price'].max() - self.products['price'].min())
        self.products['user_rating'] = (self.products['user_rating'] - self.products['user_rating'].min()) / (
                self.products['user_rating'].max() - self.products['user_rating'].min())

    @staticmethod
    def one_hot_encode(element, list):
        one_hot_encode_list = []

        for e in list:
            if element == e:
                one_hot_encode_list.append(1)
            else:
                one_hot_encode_list.append(0)
        return one_hot_encode_list

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
