from hisscube.builders import HiSSCubeConstructionDirector


class CLICommandInvoker:
    def __init__(self, args, services):
        self.args = args
        self.services = services

    def execute(self):
        if self.args.command == "ingest" or self.args.command == "update":
            build_director = HiSSCubeConstructionDirector(self.args, self.services)
            build_director.construct()

