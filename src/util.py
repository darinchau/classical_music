def is_ipython():
    try:
        __IPYTHON__ # type: ignore
        return True
    except NameError:
        return False

NATURAL = "♮"
