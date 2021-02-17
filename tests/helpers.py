def assert_list_match(list1, list2):
    """2つのリストが同じ要素を格納しているかチェックする関数"""
    assert sorted(list1) == sorted(list2)
