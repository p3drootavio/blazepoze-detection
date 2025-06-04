def check_instance(object):
    if not isinstance(object, dict):
        return False

    for key, value in object.items():
        if not (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], bool)
            and isinstance(value[1], (int, float))
            and 0 <= value[1] <= 1
        ):
            return False

    return True
