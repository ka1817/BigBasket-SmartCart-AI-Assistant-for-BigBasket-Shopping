from src.retrival_genaration import QueryRouter
if __name__ == "__main__":
    router = QueryRouter(top_k=5)
    test_query = "Do you have anything around ₹200 in cleaning supplies?"
    result = router.route(test_query)

    print("The result:", result)
