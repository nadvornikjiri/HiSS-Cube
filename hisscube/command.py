from hisscube.builders import HiSSCubeConstructionDirector


class CLICommandInvoker:
    def __init__(self, args, dependencies):
        self.args = args
        self.dependencies = dependencies

    def execute(self):
        if self.args.command == "create" or self.args.command == "update":
            build_director = HiSSCubeConstructionDirector(self.args, self.dependencies.config,
                                                          self.dependencies.serial_builders,
                                                          self.dependencies.parallel_builders)
            build_director.construct()
