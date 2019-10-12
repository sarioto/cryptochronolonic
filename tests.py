


class Tester(object):

    def __init__(self):
        return

    def get_keys(self):
        with open("./godsplan.txt") as f:
            content = f.readlines()
            content[0] = content[0][:-1]
            if (content[1][-1:] == "\n"):
                content[1] = content[1][:-1]
            return content


ts = Tester()
print(ts.get_keys())            