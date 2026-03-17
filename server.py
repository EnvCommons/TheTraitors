from openreward.environments import Server

from thetraitors import TraitorsEnvironment

if __name__ == "__main__":
    Server([TraitorsEnvironment]).run()
