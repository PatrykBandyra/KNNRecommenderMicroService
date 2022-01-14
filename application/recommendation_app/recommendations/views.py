from django.shortcuts import render
from django.views.decorators.http import require_GET
from django.http import JsonResponse
from .utils import SimpleRecommender, AdvancedRecommender, TheBestRecommender, ExperimentAB


@require_GET
def get_simple_recommendation(request, product_id):
    """
    Given product id in GET request, returns response with 5 similar recommended products.
    """
    try:
        simple_recommender = SimpleRecommender()
        neighbours = simple_recommender.get_neighbours(product_id, 5)
        data = {}

        for neighbour in neighbours:
            product = simple_recommender.products.iloc[neighbour[0]]
            data[int(product['product_id'])] = {
                'product_name': str(product['product_name']),
                'category_name': str(product['category_name']),
                'price': float(product['original_price']),
                'user_rating': float(product['original_user_rating']),
                'distance': float(neighbour[1])
            }

        return JsonResponse(data)
    except IndexError:
        return JsonResponse({'Error': 'Invalid product id!'})


@require_GET
def get_more_advanced_recommendation(request, user_id):
    """
    Given user id in GET request, returns response with 5 recommended products.
    """
    try:
        advanced_recommender = AdvancedRecommender()
        neighbours = advanced_recommender.get_recommendation(user_id, 5)
        data = {}

        for neighbour in neighbours:
            product = advanced_recommender.products.iloc[neighbour[0]]
            data[int(product['product_id'])] = {
                'product_name': str(product['product_name']),
                'category_name': str(product['category_name']),
                'price': float(product['original_price']),
                'user_rating': float(product['original_user_rating']),
                'distance': float(neighbour[1])
            }

        return JsonResponse(data)
    except IndexError:
        return JsonResponse({'Error': 'Invalid user id!'})


@require_GET
def get_best_recommendation(request, user_id):
    """
    Given user id in GET request, returns response with 5 recommended products.
    """
    try:
        best_recommender = TheBestRecommender()
        recommendations = best_recommender.get_recommendation(user_id, 5)

        data = {}
        for recommendation in recommendations:
            product = best_recommender.products.loc[best_recommender.products['product_id'] == recommendation]
            data[int(product['product_id'])] = {
                'product_name': str(product['product_name'].item()),
                'category_name': str(product['category_path'].item()),
                'price': float(product['price']),
                'user_rating': float(product['user_rating']),
            }

        return JsonResponse(data)
    except IndexError:
        return JsonResponse({'Error': 'Invalid user id!'})


@require_GET
def get_ab_experiment_by_user_ids(request):
    """
    Given list of user ids in GET request, returns response with accuracy of predictions of both models for specific
    users and predictions for users.
    User ids list example:
    http://127.0.0.1:8000/api/recommendations/ab/?id=102,103,104
    """
    try:
        user_ids = request.GET.get('id')
        user_ids = [int(id) for id in user_ids.split(',')]

        ab = ExperimentAB()
        K = 5

        data = {}

        data['accuracy_A'] = ab.get_accuracy_A_by_user_id(user_ids, K)
        data['accuracy_B'] = ab.get_accuracy_B_by_user_id(user_ids, K)
        data['A'] = {u: {} for u in user_ids}
        data['B'] = {u: {} for u in user_ids}

        for user in user_ids:
            recommendations_A = ab.get_recommendationA(user, K)
            for r in recommendations_A:
                product = ab.products.iloc[r[0]]
                data['A'][user][int(product['product_id'])] = {
                    'product_name': str(product['product_name']),
                    'category_name': str(product['category_path']),
                    'price': float(product['price']),
                    'user_rating': float(product['user_rating']),
                    'distance': float(r[1])
                }

            recommendations_B = ab.get_recommendationB(user, K)
            for r in recommendations_B:
                product = ab.products.loc[ab.products['product_id'] == r]
                data['B'][user][int(product['product_id'])] = {
                    'product_name': str(product['product_name'].item()),
                    'category_name': str(product['category_path'].item()),
                    'price': float(product['price']),
                    'user_rating': float(product['user_rating']),
                }

        return JsonResponse(data)
    except IndexError as e:
        print(e)
        return JsonResponse({'Error': 'Invalid user ids!'})


@require_GET
def get_ab_experiment(request):
    """
    Performs A/B experiment on testset and returns recommendation accuracy of both models.
    """
    try:
        ab = ExperimentAB()
        K = 5
        data = {'A': ab.get_accuracy_A(K), 'B': ab.get_accuracy_B(K)}

        return JsonResponse(data)
    except IndexError:
        return JsonResponse({'Error': 'Invalid user id!'})