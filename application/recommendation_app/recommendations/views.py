from django.shortcuts import render
from django.views.decorators.http import require_GET
from django.http import JsonResponse
from .utils import SimpleRecommender, AdvancedRecommender


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
                'user_rating': float(product['user_rating']),
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
                'user_rating': float(product['user_rating']),
                'distance': float(neighbour[1])
            }

        return JsonResponse(data)
    except IndexError:
        return JsonResponse({'Error': 'Invalid product id!'})
