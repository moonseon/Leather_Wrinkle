def judge(results):
    for r in results:
        if len(r.boxes) > 0:
            return "NG"
    return "OK"
